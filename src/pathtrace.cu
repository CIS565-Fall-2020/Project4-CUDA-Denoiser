#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "common.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

#define GBUFFER_MODE 0
#define P 0
#define N 1
#define T 2
// GBuffer for denoiser
__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);

        float timeToIntersect = gBuffer[index].t * 256.0;
        glm::vec3 norCol = glm::clamp(glm::abs(gBuffer[index].nor * 255.0f), 0.0f, 255.0f);
        glm::vec3 posCol = glm::clamp(glm::abs(gBuffer[index].pos * 20.0f), 0.0f, 255.0f);

        pbo[index].w = 0;
        #if GBUFFER_MODE == P
        pbo[index].x = posCol.x;
        pbo[index].y = posCol.y;
        pbo[index].z = posCol.z;
        #elif GBUFFER_MODE == N
        pbo[index].x = norCol.x;
        pbo[index].y = norCol.y;
        pbo[index].z = norCol.z;
        #elif GBUFFER_MODE == T
        pbo[index].x = timeToIntersect;
        pbo[index].y = timeToIntersect;
        pbo[index].z = timeToIntersect;
        #endif
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static ShadeableIntersection * dev_intersections_first_bounce = NULL;
static glm::vec2 * dev_samples = NULL;
static int numSamples;

static bool firstBounceStored = false;
static bool firstBounceUsed = false;

// Denoiser
static bool denoise = false;
static float colorWeight = 0.0f;
static float normalWeight = 0.0f;
static float positionWeight = 0.0f;
static GBufferPixel * dev_gBuffer = NULL;
static float * dev_gaussianFilter = NULL;
static float * dev_filterSum;
static glm::vec3 * dev_imageDenoised = NULL;
static int initialFilterSize = 5;
static int currFilterSize = initialFilterSize;

void setDenoise(bool val) {
    denoise = val;
}

void setFilterSize(int val) {
    currFilterSize = val;
}

void setColorWeight(float val) {
    colorWeight = val;
}

void setNormalWeight(float val) {
    normalWeight = val;
}

void setPositionWeight(float val) {
    positionWeight = val;
}


__global__ void generateGaussianValues(int filterSize, float* filter, float sigma) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < filterSize && y < filterSize) {
        int index = x + (y * filterSize);
        int center = filterSize / 2.0f;
        float exponent = -(pow((float)x - center, 2) + pow((float)y - center, 2)) / (2 * pow(sigma, 2));
        filter[index] = glm::exp(exponent) / (2 * PI * pow(sigma, 2));
    }
}

// Doing this serially; the kernel is just to avoid memset and CPU summing
__global__ void sumGaussianFilter(int filterSize, float* filter, float* sum) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (x < 1) {
        *sum = 0;
        for (int i = 0; i < filterSize * filterSize; i++) {
            *sum += filter[i];
        }
    }
}

__global__ void normalizeGaussianFilter(int filterSize, float* filter, float* sum) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < filterSize && y < filterSize) {
        int index = x + (y * filterSize);
        filter[index] /= *sum;
    }
}

void generateGaussianFilter(float sigma) {
    // Assume initial filter is less than 32 * 32
    dim3 blockSize(initialFilterSize, initialFilterSize, 1);
    generateGaussianValues << <1, blockSize >> > (initialFilterSize, dev_gaussianFilter, sigma);
    float sum = 0.0f;
    sumGaussianFilter << <1, 1 >> > (initialFilterSize, dev_gaussianFilter, dev_filterSum);
    normalizeGaussianFilter<<<1, blockSize>>> (initialFilterSize, dev_gaussianFilter, dev_filterSum);
}

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memory you need
    cudaMalloc(&dev_intersections_first_bounce, pixelcount * sizeof(ShadeableIntersection));
    
   //cudaMalloc(&dev_samples, scene->state.iterations * sizeof(glm::vec2));
    numSamples = scene->state.iterations;

    // Denoiser-related buffer
    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

    cudaMalloc(&dev_gaussianFilter, initialFilterSize * initialFilterSize * sizeof(float));

    cudaMalloc(&dev_filterSum, sizeof(float));

    cudaMalloc(&dev_imageDenoised, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_imageDenoised, 0, pixelcount * sizeof(glm::vec3));

    generateGaussianFilter(1.0f);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
  	cudaFree(dev_intersections_first_bounce);
    //cudaFree(dev_samples);
    cudaFree(dev_gBuffer);
    cudaFree(dev_gaussianFilter);
    cudaFree(dev_filterSum);
    cudaFree(dev_imageDenoised);
    checkCUDAError("pathtraceFree");
}

// number samples must be a square number
__global__ void generateStratifiedSamples(glm::vec2* samples, int sqrtSamples, int iter) {
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;
    if (indexX >= sqrtSamples || indexY >= sqrtSamples) {
        return;
    }

    int index = indexY * sqrtSamples + indexX;
    float invSqrt = 1 / (float)sqrtSamples;

    thrust::default_random_engine & rng = makeSeededRandomEngine(iter, index, 0);
    thrust::uniform_real_distribution<float> u01(0, 1);
    
    glm::vec2 sample = glm::vec2(indexX + u01(rng), indexY + u01(rng));
    sample *= invSqrt;
    samples[index] = sample;
}


/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/

// Lens effect taken from http://www.pbr-book.org/3ed-2018/Camera_Models/Projective_Camera_Models.html

#define ANTI_ALIASING 1
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, iter);
        #if ANTI_ALIASING
        // implement antialiasing by jittering the ray
        thrust::uniform_real_distribution<float> u0505(-0.5, 0.5);
        glm::vec2 point = glm::vec2(x + u0505(rng), y + u0505(rng));
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)point.x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)point.y - (float)cam.resolution.y * 0.5f)
        );
        #else 
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
        #endif

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;

        // Get a jittered (randomly generated) sample on 
        // the disk-shaped lens
        if (cam.lensRadius > 0.0f) {
            thrust::uniform_real_distribution<float> u01(0, 1);
            glm::vec3 sample = sampleDiskConcentric(glm::vec2(u01(rng), u01(rng)));
            glm::vec3 lensSample = cam.lensRadius * sample;
            glm::vec3 newOrigin = segment.ray.origin + lensSample;

            // Compute focal point
            // originally did it like the book, but that required focal_dist to be negative.
            // removing the divisor still produces a good-looking result
            glm::vec3 focalPoint = segment.ray.origin + cam.focalDist * segment.ray.direction;

            segment.ray.origin = newOrigin;
            segment.ray.direction = glm::normalize(focalPoint - newOrigin);
        }
        
		
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.

__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?
            else if (geom.type == TRIANGLE)
            {
                t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == IMPLICIT)
            {
                t = implicitSurfaceIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial (
  int iter
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
        float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        pathSegments[idx].color *= u01(rng); // apply some noise because why not
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
    }
  }
}

__global__ void shadeBSDFs(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    glm::vec2* samples,
    int numSamples
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_paths) {
        return;
    }

    PathSegment &path = pathSegments[index];
    ShadeableIntersection &intersection = shadeableIntersections[index];
    if (intersection.t > 0.0f) {
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, path.remainingBounces);
        thrust::uniform_real_distribution<float> u01(0, 1);
        Material &material = materials[intersection.materialId];
        Ray &r = path.ray;
        glm::vec3 materialColor = material.color;
        path.remainingBounces--;

        // Terminate the ray if it hits the light
        if (material.emittance > 0) {
            path.remainingBounces = 0;
            path.color *= materialColor * material.emittance;
            return;
        }

        if (path.remainingBounces == 0) {
            path.color = glm::vec3(0.0f);
            return;
        }

        // Find new ray
        glm::vec3 intersectionPoint = r.origin + intersection.t * r.direction;
        scatterRay(path, intersectionPoint, intersection.surfaceNormal, material, rng, samples, numSamples);
    }
    else {
        path.remainingBounces = 0;
        path.color = glm::vec3(0.0f);
    }
}

__global__ void generateGBuffer(
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    GBufferPixel* gBuffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection& inter = shadeableIntersections[idx];
        Ray& r = pathSegments[idx].ray;

        gBuffer[idx].t = inter.t;
        gBuffer[idx].nor = inter.surfaceNormal;
        gBuffer[idx].pos = r.origin + inter.t * r.direction;
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

__device__ __host__ int clampValue(int val, int minV, int maxV) {
    return max(minV, min(val, maxV));
}

__device__ float calculateWeight(glm::vec3 p, glm::vec3 q, float sigma) {
    float ipiq = glm::length(p - q);
    return min(glm::exp(-ipiq / (sigma * sigma)), 1.0f);
}

__device__ int intLog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

__global__ void denoiseImage(glm::ivec2 resolution,
    glm::vec3* imageDenoised,
    glm::vec3 * image,
    GBufferPixel* gBuffer,
    float * filter,
    int gaussianSize,
    int filterSize,
    float colWeight,
    float norWeight,
    float posWeight) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int radius = filterSize / 2;
        int index = x + (y * resolution.x);
        glm::vec3 average = glm::vec3(0);
        float sumFilterWeights = 0.0f;
        glm::vec3 originalColor = image[index];
        GBufferPixel originalGBuffer = gBuffer[index];

        int iterNumber = intLog2(filterSize - gaussianSize) + 1;

        // Execute iterations of A-Trous algorithm
        for (int iter = 0; iter < iterNumber; iter++) {
            // Iterate through entire kernel
            for (int xBlur = -radius; xBlur <= radius; xBlur++) {
                // check if pixel x is in range or not
                int xCoord = x + xBlur;
                xCoord = clampValue(xCoord, 0, resolution.x - 1);

                for (int yBlur = -radius; yBlur <= radius; yBlur++) {
                    // check if pixel y is in range or not
                    int yCoord = y + yBlur;
                    yCoord = clampValue(yCoord, 0, resolution.y - 1);

                    int originalIdx = xCoord + (yCoord * resolution.x);

                    int sampleX = xBlur * (iter + 1);
                    int sampleY = yBlur * (iter + 1);


                    sampleX = clampValue(x + sampleX, 0, resolution.x - 1);
                    sampleY = clampValue(y + sampleY, 0, resolution.y - 1);

                    int sampleIdx = sampleX + (sampleY * resolution.x);

                    GBufferPixel gbp = gBuffer[sampleIdx];

                    float rtWeight = calculateWeight(originalColor, image[sampleIdx], colWeight),
                        nWeight = calculateWeight(originalGBuffer.nor, gbp.nor, norWeight),
                        pWeight = calculateWeight(originalGBuffer.pos, gbp.pos, posWeight);

                    float weight = rtWeight * nWeight * pWeight;

                    // calculate gaussian weight of that pixel and add its color to average
                    int i = (xBlur + radius) * (yBlur + radius) * filterSize;
                    i = clampValue(i, 0, filterSize * filterSize - 1);
                    average += filter[i] * image[sampleIdx];
                    sumFilterWeights += weight * filter[i];
                }
            }
        }

       if (sumFilterWeights > 0) {
           imageDenoised[index] = average / sumFilterWeights;
       }
       else {
           imageDenoised[index] = image[index];
       }
    }
}

struct path_alive {
    __host__ __device__
    bool operator()(const PathSegment& p) {
        return p.remainingBounces > 0;
    }

};

struct intersection_comp {
    __host__ __device__
    bool operator()(const ShadeableIntersection &i1,
                    const ShadeableIntersection &i2) {
        return i1.materialId < i2.materialId;
    }
};
    
/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */

#define SORT_BY_MATERIAL 1
#define CACHE_FIRST_BOUNCE 1
#define USE_SAMPLES 0

using Common::PerformanceTimer;
PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
}

#define numCPUIterations -1
#define numGPUIterations -1
static float averageCPUIterTime = 0.0f;
static float averageGPUIterTime = 0.0f;
void pathtrace(uchar4 *pbo, int frame, int iter) {
    #if USE_SAMPLES
    if (iter <= 1) {
        int sqrtIter = sqrt(numSamples);
        dim3 blockDimSamples(32, 32, 1);
        int numBlocksSamples = ceil((float)sqrtIter / blockDimSamples.x);
        generateStratifiedSamples << < numBlocksSamples, blockDimSamples >> > (dev_samples, sqrtIter, iter);
    }
    #endif

    if (iter <= numCPUIterations) {
        timer().startCpuTimer();
    }

    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // Empty gbuffer
    cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

    #if CACHE_FIRST_BOUNCE && ANTI_ALIASING == 0
    firstBounceUsed = false;
    if (iter <= 1) {
        cudaMemset(dev_intersections_first_bounce, 0, pixelcount * sizeof(ShadeableIntersection));
        firstBounceStored = false;
    }
    #endif

    bool iterationComplete = false;

	while (!iterationComplete) {

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// tracing
	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
    #if CACHE_FIRST_BOUNCE && ANTI_ALIASING == 0
    if ((iter <= 1 && !firstBounceStored) || firstBounceUsed) {
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_intersections
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;
    }
    else if (iter > 1 && firstBounceStored) {
        cudaMemcpy(dev_intersections, dev_intersections_first_bounce, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        firstBounceUsed = true;
    }
    else {
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_intersections
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;
    }
    #else
    computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
        depth
        , num_paths
        , dev_paths
        , dev_geoms
        , hst_scene->geoms.size()
        , dev_intersections
        );
    checkCUDAError("trace one bounce");
    cudaDeviceSynchronize();
    depth++;
    #endif


    if (depth == 1) {
        generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (pixelcount, dev_intersections, dev_paths, dev_gBuffer);
    }

	// Shade path segments based on intersections and generate new rays by
    // evaluating the BSDF.

    shadeBSDFs << <numblocksPathSegmentTracing, blockSize1d >> > (
        iter,
        num_paths,
        dev_intersections,
        dev_paths,
        dev_materials,
        dev_samples,
        numSamples);


    #if CACHE_FIRST_BOUNCE && ANTI_ALIASING == 0
    if (!firstBounceStored) {
        cudaMemcpy(dev_intersections_first_bounce, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        firstBounceStored = true;
    }
    #endif
    
    #if SORT_BY_MATERIAL
        thrust::stable_sort_by_key(thrust::device,
                                   dev_intersections,
                                   dev_intersections + num_paths,
                                   dev_paths, intersection_comp());
    #endif
    
    // STREAM COMPACTION
    dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_path_end, path_alive());
    num_paths = dev_path_end - dev_paths;
    iterationComplete = num_paths == 0;

	}

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////
    
    timer().startGpuTimer();
    // if denoising, run kernels that take both the raw pathtraced resultand the gbuffer, and put the result in the "pbo" from opengl
    denoiseImage << <blocksPerGrid2d, blockSize2d >> > (cam.resolution,
        dev_imageDenoised,
        dev_image,
        dev_gBuffer,
        dev_gaussianFilter,
        initialFilterSize,
        currFilterSize,
        colorWeight,
        normalWeight,
        positionWeight);
    
    timer().endGpuTimer();

    if (denoise) {
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_imageDenoised);
        // Retrieve image from GPU
    }
    else {
        // Send results to OpenGL buffer for rendering
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
    }

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
    
    if (iter <= numCPUIterations) {
        timer().endCpuTimer();
        averageCPUIterTime += timer().getCpuElapsedTimeForPreviousOperation();
        if (iter == numCPUIterations) {
            averageCPUIterTime /= numCPUIterations;
            std::cout << "Average CPU iteration time: " << averageCPUIterTime << "ms " << std::endl;
            averageCPUIterTime = 0.0f;
        }
    }
    
    if (iter <= numGPUIterations) {
        averageGPUIterTime += timer().getGpuElapsedTimeForPreviousOperation();
        if (iter == numGPUIterations) {
            averageGPUIterTime /= numGPUIterations;
            std::cout << "Average GPU iteration time: " << averageGPUIterTime << "ms " << std::endl;
            averageGPUIterTime = 0.0f;
        }
    }
}


// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
    gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
}

void showImage(uchar4* pbo, int iter) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        cam.resolution.x * cam.resolution.y * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}

void showDenoisedImage(uchar4* pbo, int iter) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_imageDenoised);
    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_imageDenoised,
        cam.resolution.x * cam.resolution.y * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}
