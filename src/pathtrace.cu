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
#define VISUALIZE_POS 0
#define DEBUG 0

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

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        float timeToIntersect = gBuffer[index].t * 256.0;

        pbo[index].w = 0;
        /* GIVEN DUMMY GBUFFER
        pbo[index].x = timeToIntersect;
        pbo[index].y = timeToIntersect;
        pbo[index].z = timeToIntersect;
        */

        // this is to display the normals gbuffer
        glm::vec3 temp_nor = glm::normalize(glm::abs(gBuffer[index].nor)) * 255.f;
        pbo[index].x = temp_nor[0];
        pbo[index].y = temp_nor[1];
        pbo[index].z = temp_nor[2];

        // this is to display the position gbuffer
#if VISUALIZE_POS
        glm::vec3 temp_pos = glm::clamp(glm::abs(gBuffer[index].pos * 28.f), 0.f, 255.f);
        pbo[index].x = temp_pos[0];
        pbo[index].y = temp_pos[1];
        pbo[index].z = temp_pos[2];
#endif
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
static glm::vec3* dev_image_swap = NULL; // this is the buffer we need to write noised pixel data 
static glm::vec3* dev_image_raw = NULL; // this is the buffer that holds the raw raytraced results of each iteration

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

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_image_swap, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image_swap, 0, pixelcount * sizeof(glm::vec3));
    
    cudaMalloc(&dev_image_raw, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image_raw, 0, pixelcount * sizeof(glm::vec3));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    cudaFree(dev_gBuffer);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_image_swap);
    cudaFree(dev_image_raw);
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
            intersections[path_index].position = intersect_point;
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
  GBufferPixel* gBuffer) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    gBuffer[idx].t = shadeableIntersections[idx].t;
    gBuffer[idx].nor = shadeableIntersections[idx].surfaceNormal;
    gBuffer[idx].pos = shadeableIntersections[idx].position;
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

// Add the current iteration's output to a tempimage
__global__ void tempGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] = iterationPath.color;
    }
}

// Add the denoised result from this iteration to the overall image 
__global__ void finalDenoiseGather(glm::vec3* img_denoised, glm::vec3* img_overall, Camera cam) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        img_overall[index] += img_denoised[index];
    }
}

// A-Trous filter 
__global__ void atrous_filter(
    glm::vec3* img_raw,
    glm::vec3* img_swap,
    GBufferPixel* gBuffer,
    Camera cam,
    float color_weight,
    float nor_weight, 
    float pos_weight,
    int step_width
) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        // given to us on the top of page 4
        float kernel[5] = { 1.0 / 16.0, 
                       1.0 / 4.0, 
                       3.0 / 8.0, 
                       1.0 / 4.0, 
                       1.0 / 16.0 };
        // we need to initialize the variables as seen in the sample code
        glm::vec3 sum = glm::vec3(0.f);
        glm::vec3 cval = img_raw[index];
        glm::vec3 nval = gBuffer[index].nor;
        glm::vec3 pval = gBuffer[index].pos;

        float cum_w = 0.f;
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                glm::ivec2 uv = glm::ivec2(x + (i - 2) * step_width,
                                           y + (j - 2) * step_width);
                uv = glm::clamp(uv, glm::ivec2(0, 0), cam.resolution - glm::ivec2(1, 1));
                int uv_1d = uv.x + (uv.y * cam.resolution.x);
                
                // calculate color weight of original image
                glm::vec3 ctmp = img_raw[uv_1d];
                glm::vec3 t = (cval - ctmp);
                float dist2 = glm::dot(t, t);
                float c_w = glm::min(glm::exp(-(dist2) / color_weight), 1.f);

                // calculate normal weight based on gbuffer
                glm::vec3 ntmp = gBuffer[uv_1d].nor;
                t = nval - ntmp;
                dist2 = glm::max(glm::dot(t, t) / (step_width * step_width), 0.f);
                float n_w = glm::min(glm::exp(-(dist2) / nor_weight), 1.f);

                // calculate the position weight based on gbuffer
                glm::vec3 ptmp = gBuffer[uv_1d].pos;
                t = pval - ptmp;
                dist2 = glm::dot(t, t);
                float p_w = glm::min(glm::exp(-(dist2) / pos_weight), 1.f);
                
                float weight = c_w * n_w * p_w; // combine them to essentially do edge detection
                sum += ctmp * weight * kernel[i] * kernel[j];
                cum_w += weight * kernel[i] * kernel[j];
            }
        }
        img_swap[index] = sum / cum_w;
    }
}

//if denoising, run kernels that take both the raw pathtraced result
//and the gbuffer, and put the result in the "pbo" from opengl
void denoise_filter(int iter, int ui_filterSize, float ui_colorWeight,
    float ui_normalWeight, float ui_positionWeight) {
#if DEBUG
    cout << "calling denoise filter on iteration: " << iter << endl;
#endif
    // Do similar set up as pathtrace() ====================
    const Camera& cam = hst_scene->state.camera;

    // 2D block for generating ray from camera (or in this case, the filter)
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // end of set up =======================================

    // Writing out thought process: 
    // We need a kernel that takes in 
    //    1) The image buffer with all teh info post path tracing
    //    2) A pingpong buffer according to Piazza post @212, since we can't read and write to same buffer
    //    3) Gbuffer with the nor and pos
    //    4) Weights specified in UI (c_phi, n_phi, p_phi)
    //    5) step width
    //    6) the b spline interpolation kernel (h)
    // This kernel is called in the same way as generate ray from camera... aka a 2d grid
    // .... Need to also pass in camera to do checks
    // .... I guess we could also do a 1d grid but... 2d makes more sense
    // We also need need to double the step width every time, as seen in the recitation slides
    // The point of this kernel is to repeatedly sample from more distant neighbors and adding them 
    // into your current pixel, creating a blur. We need the weights to factor in the edges. 

    int step_width = 1; // we start by sampling the pixels immediately next to you
    float color_weight = ui_colorWeight;

    // To get number of iterations, we have to take the filter size and / 2. 
    // This gives us the number of pixels on either side. 
    // then we divide by 2 again, because in a 5x5, there are 2 pixels on either side. 
    // we then take the logbase2 of that to get # iterations
    int blur_iterations = ceil(glm::log2((ui_filterSize) / 4.f));
#if DEBUG 
    cout << "number of blur iterations: " << blur_iterations << endl;
#endif 

    for (int i = 0; i < blur_iterations; i++) {
        atrous_filter << <blocksPerGrid2d, blockSize2d >> > (
            dev_image_raw,
            dev_image_swap,
            dev_gBuffer,
            cam,
            color_weight,
            ui_normalWeight, 
            ui_positionWeight,
            step_width
            );

        // double the step width every time. This is seen from Figure 3 of the paper.
        step_width *= 2;
        // also seen in recitation slides, the outer pixels weight half as less.
        color_weight *= 0.5f;

        // the denoised result is in dev_image_swap, because we are writing to there in the kernel above
        // we need to read from dev_image_raw, so we will swap the two
        std::swap(dev_image_raw, dev_image_swap); // pingpong the buffers back and forth
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter, bool denoise, 
    int ui_filterSize, float ui_colorWeight,
    float ui_normalWeight, float ui_positionWeight) {
#if DEBUG
    cout << "calling path trace on iteration: " << iter << endl;
#endif
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
    generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_paths, dev_gBuffer);
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
  //     * if not denoising, add this iteration's results to the image
  if (!denoise) {
      finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);
  }
  //     * if denoising, run kernels that take both the raw pathtraced result and the gbuffer, 
  //       and put the result in the "pbo" from opengl
  else {
      tempGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image_raw, dev_paths);
      denoise_filter(iter, ui_filterSize, ui_colorWeight,
          ui_normalWeight, ui_positionWeight);
      finalDenoiseGather << <blocksPerGrid2d, blockSize2d >> > (dev_image_raw, dev_image, cam);
  }

    ///////////////////////////////////////////////////////////////////////////

    // CHECKITOUT: use dev_image as reference if you want to implement saving denoised images.
    // Otherwise, screenshots are also acceptable.
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
    gbufferToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_gBuffer);
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
