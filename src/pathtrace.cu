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

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer, int bufferMode) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);

        GBufferType bufType = (GBufferType)bufferMode;
        if (bufType == POS) {
            // Use intersection position
            glm::vec3 posScale = gBuffer[index].pos * 25.f; // arbitary scale factor
            posScale = glm::clamp(glm::vec3(glm::abs(posScale.x), glm::abs(posScale.y), glm::abs(posScale.z)), 0.f, 255.f);
            pbo[index].w = 0;
            pbo[index].x = posScale.x;
            pbo[index].y = posScale.y;
            pbo[index].z = posScale.z;
        }
        else if (bufType == NORM) {
            // Use intersection normal
            glm::vec3 norm = gBuffer[index].nor;
            if (glm::length(norm) > 0.f) {
                norm = glm::vec3(0.5f * norm.x + 0.5f, 0.5f * norm.y + 0.5f, 0.5f * norm.z + 0.5f) * 255.f;
            }
            pbo[index].w = 0;
            pbo[index].x = norm.x;
            pbo[index].y = norm.y;
            pbo[index].z = norm.z;
        }
        else if (bufType == T) {
            float timeToIntersect = gBuffer[index].t * 256.0;
            pbo[index].w = 0;
            pbo[index].x = timeToIntersect;
            pbo[index].y = timeToIntersect;
            pbo[index].z = timeToIntersect;
        }
        else if (bufType == COLOR) {
            glm::vec3 color = gBuffer[index].color * 255.f;
            pbo[index].w = 0;
            pbo[index].x = color.x;
            pbo[index].y = color.y;
            pbo[index].z = color.z;
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
// Ping-pong buffers for denoising
static glm::vec3* dev_denoised_image_input = NULL; // stored denoised image colors input
static glm::vec3* dev_denoised_image_output = NULL; // stored denoised image colors output
static float* dev_filter = NULL; // blur filter kernel
static glm::vec2* dev_offset = NULL; // filter offsets
static std::vector<float> hst_filter{ 0.003765f ,0.015019f, 0.023792f, 0.015019f, 0.003765f,
                                      0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f,
                                      0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f,
                                      0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f,
                                      0.003765f , 0.015019f, 0.023792f, 0.015019f, 0.003765f };
static std::vector<glm::vec2> hst_offset;

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

    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

    cudaMalloc(&dev_denoised_image_input, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_image_input, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_denoised_image_output, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_image_output, 0, pixelcount * sizeof(glm::vec3));

    // Setup the filter kernel and offsets
    // Assume filter size is 25 for now - potentially change this later!
    for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
            hst_offset.push_back(glm::vec2(j, i));
        }
    }

    cudaMalloc(&dev_filter, hst_filter.size() * sizeof(float));
    cudaMemcpy(dev_filter, hst_filter.data(), hst_filter.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_offset, hst_offset.size() * sizeof(glm::vec2));
    cudaMemcpy(dev_offset, hst_offset.data(), hst_offset.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    cudaFree(dev_gBuffer);
    cudaFree(dev_denoised_image_input);
    cudaFree(dev_denoised_image_output);
    hst_offset.clear();
    cudaFree(dev_filter);
    cudaFree(dev_offset);

    checkCUDAError("pathtraceFree");
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

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

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

__global__ void shadeSimpleMaterials (
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
    PathSegment segment = pathSegments[idx];
    if (segment.remainingBounces == 0) {
      return;
    }

    if (intersection.t > 0.0f) { // if the intersection exists...
      segment.remainingBounces--;
      // Set up the RNG
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, segment.remainingBounces);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        segment.color *= (materialColor * material.emittance);
        segment.remainingBounces = 0;
      }
      else {
        segment.color *= materialColor;
        glm::vec3 intersectPos = intersection.t * segment.ray.direction + segment.ray.origin;
        scatterRay(segment, intersectPos, intersection.surfaceNormal, material, rng);
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      segment.color = glm::vec3(0.0f);
      segment.remainingBounces = 0;
    }

    pathSegments[idx] = segment;
  }
}

__global__ void generateGBuffer (
  int num_paths,
  ShadeableIntersection* shadeableIntersections,
  PathSegment* pathSegments,
  GBufferPixel* gBuffer,
  Material* materials) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    // Fill gBuffer intersection t data
    float t = shadeableIntersections[idx].t;
    GBufferPixel temp;

    // Fill gBuffer intersection position data
    if (t > 0.f) {
        temp.pos = getPointOnRay(pathSegments[idx].ray, t);
        temp.nor = shadeableIntersections[idx].surfaceNormal;
        temp.color = materials[shadeableIntersections[idx].materialId].color;
    }
    temp.t = t;
    gBuffer[idx] = temp;
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

__global__ void computeDenoisedImage(
    int nPaths,
    glm::vec3* denoisedInput,
    glm::vec3* denoisedOutput,
    GBufferPixel* gBuffer,
    int filterSize,
    float* kernel,
    glm::vec2* offset,
    float cPhi,
    float pPhi,
    float nPhi,
    float stepWidth,
    glm::vec2 resolution) 
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < nPaths) {
        glm::vec3 colorSum(0.f); // store color sum here
        float weightSum = 0.f;
        // get current pixel's 2D coordinates
        int x = index % (int)resolution.y;
        int y = index / (int)resolution.y;
        // loop over all the pixels in the filter
        for (int i = 0; i < filterSize; ++i) {
            // compute filter pixel coordinate
            int pix_x = x + offset[i].x * stepWidth;
            int pix_y = y + offset[i].y * stepWidth;

            if (pix_x < 0 || pix_x >= resolution.x || pix_y < 0 || pix_y >= resolution.y) continue; // skip out of bound coordinates

            int pix_idx = pix_x + (pix_y * resolution.x);

            // Compute c_w
            float d_x = denoisedInput[index].x;
            float d_y = denoisedInput[index].y;
            float d_z = denoisedInput[index].z;
            float dp_x = denoisedInput[pix_idx].x;
            float dp_y = denoisedInput[pix_idx].y;
            float dp_z = denoisedInput[pix_idx].z;
            glm::vec3 diff = glm::vec3(d_x, d_y, d_z) - glm::vec3(dp_x, dp_y, dp_z);
            float len = glm::dot(diff, diff);
            float c_w = glm::min(glm::exp(-(len) / cPhi), 1.f);

            // Compute n_w
            diff = gBuffer[index].nor - gBuffer[pix_idx].nor;
            len = glm::dot(diff, diff) / (stepWidth * stepWidth);
            float n_w = glm::min(glm::exp(-(len) / nPhi), 1.f);

            // Compute p_w
            diff = gBuffer[index].pos - gBuffer[pix_idx].pos;
            len = glm::dot(diff, diff);
            float p_w = glm::min(glm::exp(-(len) / pPhi), 1.f);

            // weight
            float w = c_w * n_w * p_w;
            colorSum += denoisedInput[pix_idx] * w * kernel[i];
            weightSum += w * kernel[i];

            // do regular gaussian blur
            //colorSum += (kernel[i] * denoisedInput[pix_idx]);
            //weightSum += kernel[i];
        }

        // write the output
        denoisedOutput[index] = (colorSum / weightSum);
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter) {
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

    // Pathtracing Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * NEW: For the first depth, generate geometry buffers (gbuffers)
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally:
    //     * if not denoising, add this iteration's results to the image
    //     * TODO: if denoising, run kernels that take both the raw pathtraced result and the gbuffer, and put the result in the "pbo" from opengl

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  // Empty gbuffer
  cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

  bool iterationComplete = false;
	while (!iterationComplete) {

	// tracing
	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
	computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
		depth
		, num_paths
		, dev_paths
		, dev_geoms
		, hst_scene->geoms.size()
		, dev_intersections
		);
	checkCUDAError("trace one bounce");
	cudaDeviceSynchronize();

  if (depth == 0) {
    generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_paths, dev_gBuffer, dev_materials);
  }

	depth++;

  shadeSimpleMaterials<<<numblocksPathSegmentTracing, blockSize1d>>> (
    iter,
    num_paths,
    dev_intersections,
    dev_paths,
    dev_materials
  );
  iterationComplete = depth == traceDepth;
	}

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // CHECKITOUT: use dev_image as reference if you want to implement saving denoised images.
    // Otherwise, screenshots are also acceptable.
    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

void denoiseImage(int num_iters, float c_phi, float n_phi, float p_phi) {
    
    const Camera& cam = hst_scene->state.camera;
    const int blockSize1d = 128;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    dim3 numblocksDenoising = (pixelcount + blockSize1d - 1) / blockSize1d;
    // copy current image to dev_denoised_image_input
    cudaMemcpy(dev_denoised_image_input, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    checkCUDAError("copy image to denoise buffer");

    // denoise image for given number of iterations
    for (int i = 0; i < num_iters; ++i) {
        computeDenoisedImage << <numblocksDenoising, blockSize1d >> > (
                pixelcount,
                dev_denoised_image_input,
                dev_denoised_image_output,
                dev_gBuffer,
                hst_filter.size(),
                dev_filter,
                dev_offset,
                c_phi,
                p_phi,
                n_phi,
                glm::pow(2, i),
                cam.resolution);
        checkCUDAError("denoise image");
        // ping-pong buffers
        std::swap(dev_denoised_image_input, dev_denoised_image_output);
        checkCUDAError("swap buffers");
    }
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo, int buffer_type) {
    const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
    gbufferToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_gBuffer, buffer_type);
}

void showImage(uchar4* pbo, int iter, bool denoised) {
const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    if (denoised) {
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_denoised_image_input);
    }
    else {
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
    }
}
