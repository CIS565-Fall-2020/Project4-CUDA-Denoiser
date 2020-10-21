#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

using utilTimer::PerformanceTimer;

#define ERRORCHECK 1
#define CACHE_BOUNCE 1
#define MATERIAL_SORT 1
#define DEPTH_OF_FIELD 0
#define ANTIALIASING 1
#define GPU_TIMER 1

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

PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
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

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer, GBufferMode mode) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);

        if(mode == POSITION)
        {
            glm::vec3 posValue = glm::abs(gBuffer[index].pos) * 25.f;
            posValue = glm::clamp(posValue, 0.f, 255.f);
            pbo[index].w = 0;
            pbo[index].x = posValue.x;
            pbo[index].y = posValue.y;
            pbo[index].z = posValue.z;
        }
        else if(mode == NORMAL)
        {
            glm::vec3 norValue = glm::abs(gBuffer[index].nor) * 255.f;
            norValue = glm::clamp(norValue, 0.f, 255.f);
            pbo[index].w = 0;
            pbo[index].x = norValue.x;
            pbo[index].y = norValue.y;
            pbo[index].z = norValue.z;
        }
        else if(mode == DUMMY)
        {
            float timeToIntersect = gBuffer[index].t * 256.0;
            pbo[index].w = 0;
            pbo[index].x = timeToIntersect;
            pbo[index].y = timeToIntersect;
            pbo[index].z = timeToIntersect;
        }
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static PathSegment * dev_cache_paths = NULL;
static ShadeableIntersection * dev_cache_intersections = NULL;
static Triangle* dev_triangles = NULL;
static int* dev_idxOfEachMesh = NULL;
static int* dev_endIdxOfEachMesh = NULL;
static glm::vec3* dev_denoise = NULL;
static glm::vec3* dev_denoise2 = NULL;

static int mesh_size = 0;
static int triangle_size = 0;
static std::vector<int> indexOffset;

cudaEvent_t start, stop;
float totalTime = 0.f;
bool timerStart = true;

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

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_cache_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

    cudaMalloc(&dev_denoise, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoise, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_denoise2, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoise2, 0, pixelcount * sizeof(glm::vec3));


    mesh_size = scene->meshes.size();
    //cout << mesh_size << endl;
    triangle_size = scene->totalTriangles;
    cudaMalloc(&dev_triangles, triangle_size * sizeof(Triangle));
    for (int i = 0; i < mesh_size; i++) {
        int triangle_size_per = scene->meshes[i].size();
        int offset = scene->idxOfEachMesh[i];
        cudaMemcpy(dev_triangles + offset, scene->meshes[i].data(), triangle_size_per * sizeof(Triangle), cudaMemcpyHostToDevice);
        //cout << triangle_size_per << endl;
    }

    cudaMalloc(&dev_idxOfEachMesh, mesh_size * sizeof(int));
    cudaMemcpy(dev_idxOfEachMesh, scene->idxOfEachMesh.data(), mesh_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_endIdxOfEachMesh, mesh_size * sizeof(int));
    cudaMemcpy(dev_endIdxOfEachMesh, scene->endIdxOfEachMesh.data(), mesh_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
	cudaEventCreate(&stop);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_cache_paths);
    cudaFree(dev_cache_intersections);
    cudaFree(dev_triangles);
    cudaFree(dev_idxOfEachMesh);
    cudaFree(dev_endIdxOfEachMesh);
    cudaFree(dev_gBuffer);
    cudaFree(dev_denoise);
    cudaFree(dev_denoise2);

    checkCUDAError("pathtraceFree");
}

__host__ __device__
glm::vec3 squareToDiskUniform(const glm::vec2& sample)
{
	float r = sqrt(sample.x);
	float theta = 2 * PI * sample.y;
	float x = r * cos(theta);
	float y = r * sin(theta);
	return glm::vec3(x, y, 0.f);
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {

		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

		// TODO: implement antialiasing by jittering the ray
#if ANTIALIASING == 1 && CACHE_BOUNCE == 0

    segment.ray.direction = glm::normalize(cam.view
        - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u01(rng) - 0.5f)
        - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u01(rng) - 0.5f)
    );

#else
	segment.ray.direction = glm::normalize(cam.view
		- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
		- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);

#endif //ANTIALIASING



#if DEPTH_OF_FIELD == 1

    float discRadius = 1.f;
    float focalDistance = 7.f;   // distance between the projection point and the plane where everything is in perfect focus

    glm::vec2 sample = glm::vec2(u01(rng), u01(rng));
    glm::vec3 pLens = discRadius * squareToDiskUniform(sample);

    float ft = focalDistance / glm::abs(segment.ray.direction.z);
    glm::vec3 pFocus = segment.ray.origin + ft * segment.ray.direction;

    segment.ray.origin += glm::vec3(pLens.x, pLens.y, 0.f);
    segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);

#endif  //DEPTH_OF_FEILD

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
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
    , Triangle *triangles
	, int geoms_size
	, ShadeableIntersection * intersections
    , int* idxOfEachMesh
    , int* endIdxOfEachMesh
    , int iter
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
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, path_index, 0);
                thrust::uniform_real_distribution<float> u01(0, 1);
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, u01(rng));
			}
            else if (geom.type == MESH) {
                int startIdx = idxOfEachMesh[geom.meshIdx];
                int endIdx = endIdxOfEachMesh[geom.meshIdx];
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, path_index, 0);
                thrust::uniform_real_distribution<float> u01(0, 1);
                 t = meshIntersectionTest(geom, triangles, pathSegment.ray, tmp_intersect, tmp_normal, outside, startIdx, endIdx, u01(rng));  
             }
			// TODO: add more intersection tests here... triangle? metaball? CSG?

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

__global__ void shadeRealMaterial (
  int iter
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_paths) {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            if(material.emittance > 0.f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = -1;
            }
            else
            {
                glm::vec3 intersectPoint = getPointOnRay(pathSegments[idx].ray, intersection.t);
                glm::vec3 normal = intersection.surfaceNormal;
                scatterRay(pathSegments[idx], intersectPoint, normal, material, rng);
            }
        } 
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = -1;
        }
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

__global__ void generateGBuffer (
  int num_paths,
  ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
  GBufferPixel* gBuffer) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    gBuffer[idx].t = shadeableIntersections[idx].t;
    gBuffer[idx].pos = getPointOnRay(pathSegments[idx].ray, shadeableIntersections[idx].t);
    gBuffer[idx].nor = shadeableIntersections[idx].surfaceNormal;
  }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
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

    // TODO: perform one iteration of path tracing

    //timer().startGpuTimer();

#if GPU_TIMER == 1
	cudaEventRecord(start);
#endif // GPU_TIMER

#if CACHE_BOUNCE == 1

    if(iter == 1)
    {
        generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
        checkCUDAError("generate camera ray");

        cudaMemcpy(dev_cache_paths, dev_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
    }
    else 
    {
        cudaMemcpy(dev_paths, dev_cache_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
    }
#else
    generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

#endif //CACHE_BOUNCE

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks    

      // Empty gbuffer
  cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

    bool iterationComplete = false;
	while (!iterationComplete) {

        // clean shading chunks
	    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	    // tracing
	    dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;



    #if CACHE_BOUNCE == 1
        if(depth == 0)
        {
            if(iter == 1)
            {   
                computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
		            depth, num_paths, dev_paths, dev_geoms, dev_triangles
		            , hst_scene->geoms.size(), dev_intersections
                    , dev_idxOfEachMesh, dev_endIdxOfEachMesh, iter);

	            checkCUDAError("trace one bounce");
	            //cudaDeviceSynchronize();
                generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_paths, dev_gBuffer);
	            depth++;
               
                cudaMemcpy(dev_cache_intersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
                cudaDeviceSynchronize();
            }
            else 
            {        
                cudaMemcpy(dev_intersections, dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
                generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_paths, dev_gBuffer);

                depth++;
                cudaDeviceSynchronize();
            }
        }
        else
        {
            computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
		        depth, num_paths, dev_paths, dev_geoms, dev_triangles
		        , hst_scene->geoms.size(), dev_intersections
                , dev_idxOfEachMesh, dev_endIdxOfEachMesh, iter);

	        checkCUDAError("trace one bounce");
	        cudaDeviceSynchronize();
	        depth++;
        }
    #else
        computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
		    depth, num_paths, dev_paths, dev_geoms, dev_triangles
		    , hst_scene->geoms.size(), dev_intersections
            , dev_idxOfEachMesh, dev_endIdxOfEachMesh, iter);

	    checkCUDAError("trace one bounce");
	    cudaDeviceSynchronize();
        if (depth == 0) {
            generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_paths, dev_gBuffer);
        }
	    depth++;
    #endif //CACHE_BOUNCE
    

    #if MATERIAL_SORT == 1
        // Sort the rays/path segments
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, material_sort());
    #endif  //MATERIAL_SORT


	    // TODO:
	    // --- Shading Stage ---
	    // Shade path segments based on intersections and generate new rays by
      // evaluating the BSDF.
      // Start off with just a big kernel that handles all the different
      // materials you have in the scenefile.
      // TODO: compare between directly shading the path segments and shading
      // path segments that have been reshuffled to be contiguous in memory.



      shadeRealMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
        iter,
        num_paths,
        dev_intersections,
        dev_paths,
        dev_materials
      );

        // Stream compact away all of the terminated paths
        //dev_path_end = thrust::remove_if(thrust::device, dev_paths, dev_paths + num_paths, isTerminated());

        dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, isContinuing());
        num_paths = dev_path_end - dev_paths;

        if(num_paths <= 0 || depth >= traceDepth)
        {
            iterationComplete = true;
        }
    }
    //timer().endGpuTimer();

#if GPU_TIMER == 1
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float t;
	cudaEventElapsedTime(&t, start, stop);
	totalTime += t;
	if (timerStart && iter > 50) {
		std::cout << " time per iteration is: " << totalTime / iter << " ms" <<std::endl;
		timerStart = false;
	}
#endif // GPU_TIMER

    //printElapsedTime(timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo) {
    const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
    gbufferToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_gBuffer, GBufferMode::NORMAL);
}


void showImage(uchar4* pbo, int iter) {
const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
}

__global__ void denoisePerIter(
    glm::ivec2 resolution, 
    int stepWidth,
    GBufferPixel* gBuffer, 
    glm::vec3* denoise,
    glm::vec3* denoise2,
    float c_phi,
    float n_phi,
    float p_phi,
    int filterSize)
{   
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int idx = y *  resolution.x + x;

    if(x < resolution.x && y < resolution.y)
    {
        float kernel[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
        glm::vec3 sum = glm::vec3(0.f);
        //glm::vec2 step = glm::vec2(1.f / (float)resolution.x, 1.f / (float)resolution.y);
        glm::vec3 cval = denoise[idx];
        glm::vec3 nval = gBuffer[idx].nor;
        glm::vec3 pval = gBuffer[idx].pos;

        float sum_w = 0.f;
        for(int i = 0; i < 5; i++)
        {
            for(int j = 0; j < 5; j++)
            {
                glm::ivec2 coor = glm::ivec2(x, y) + glm::ivec2((i - 2) * stepWidth, (j - 2) * stepWidth);
                coor = glm::clamp(coor, glm::ivec2(0, 0), resolution - glm::ivec2(1, 1));
                int index = coor.y * resolution.x + coor.x;

                glm::vec3 ctmp = denoise[index];
                glm::vec3 t = cval - ctmp;
                float dist2 = glm::dot(t, t);
                float c_w = glm::min(glm::exp(-dist2 / c_phi), 1.f);

                glm::vec3 ntmp = gBuffer[index].nor;
                t = ntmp - nval;
                dist2 = glm::max(glm::dot(t, t) / (stepWidth, stepWidth), 0.f);
                float n_w = glm::min(glm::exp(-dist2 / n_phi), 1.f);

                glm::vec3 ptmp = gBuffer[index].pos;
				t = ptmp - pval;
				dist2 = glm::dot(t, t);
				float p_w = glm::min(glm::exp(-dist2 / p_phi), 1.f);

				float weight = c_w * n_w * p_w;
				sum += ctmp * weight * kernel[i] * kernel[j];
				sum_w += weight * kernel[i] * kernel[j];
            }
        }
        denoise2[idx] = sum / sum_w;
    }
}

__global__ void processImage(glm::vec3 *image, glm::vec3 *denoise, glm::ivec2 resolution, int iter)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];	
		denoise[index].x = glm::clamp(pix.x / iter, 0.f, 1.f);
		denoise[index].y = glm::clamp(pix.y / iter, 0.f, 1.f);
		denoise[index].z = glm::clamp(pix.z / iter, 0.f, 1.f);
	}
}

void denoise(uchar4* pbo, int iter, float c_phi, float n_phi, float p_phi, int filterSize)
{
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    dim3 numblocksDenoising(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    //cudaMemcpy(dev_denoise, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    processImage << < numblocksDenoising, blockSize2d >> > (dev_image, dev_denoise, cam.resolution, iter);

    for(int i = 0; i < iter; ++i) 
    {
        int stepWidth = 1 << i;
        denoisePerIter <<<numblocksDenoising, blockSize2d>>> (cam.resolution, stepWidth, dev_gBuffer, dev_denoise, dev_denoise2, c_phi, n_phi, p_phi, filterSize);

        std::swap(dev_denoise, dev_denoise2);
        checkCUDAError("swap buffers");
    }
    sendImageToPBO  << <numblocksDenoising, blockSize2d >> > (pbo, cam.resolution, 1, dev_denoise);
}
