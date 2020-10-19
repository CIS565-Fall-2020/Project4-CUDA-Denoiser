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
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
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
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer, glm::vec3* bufferImage) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);

		// this is white
		pbo[index].x = 255;
		pbo[index].y = 255;
		pbo[index].z = 255;
		// this is black
		// these bottom lines cause a warning saying that the integer was
		// truncated. this makes since because the extent for a uchar is 255
		// so the compiler will truncate or dispose of all of the bits used to
		// express values over 256
		pbo[index].x = 256;
		pbo[index].y = 256;
		pbo[index].z = 256;

		float timeToIntersect = gBuffer[index].t * 256.0;

		pbo[index].w = 0;
		pbo[index].x = timeToIntersect;
		pbo[index].y = timeToIntersect;
		pbo[index].z = timeToIntersect;

		glm::vec3 c = gBuffer[index].c * 256.f;
		pbo[index].x = c.x;
		pbo[index].y = c.y;
		pbo[index].z = c.z;

		/*glm::vec3 n = (gBuffer[index].n + 1.f) / 2.f * 256.f;
		pbo[index].x = n.x;
		pbo[index].y = n.y;
		pbo[index].z = n.z;*/

		glm::vec3 p = gBuffer[index].p * 256.f;
		pbo[index].x = p.x;
		pbo[index].y = p.y;
		pbo[index].z = p.z;

		// i think that because pbo is unmapped after it is assigned values
		// the pointer will be invalid and will not persist when i try to
		// copy its contents later on in the code
		bufferImage[index].x = pbo[index].x;
		bufferImage[index].y = pbo[index].y;
		bufferImage[index].z = pbo[index].z;
	}
}

static Scene* hst_scene = NULL;
static glm::vec3* dev_image = NULL;
static glm::vec3* dev_bufferImage = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
static glm::ivec2* dev_offset = NULL;
static float* dev_kernel = NULL;
static int lvlimit = 0;
static int kernelWidth = 0;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene* scene) {
	hst_scene = scene;
	const Camera& cam = hst_scene->state.camera;
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
	cudaMalloc(&dev_bufferImage, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_bufferImage, 0, pixelcount * sizeof(glm::vec3));

	cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

	// these will probably need to be arguments into this function from the ui
	lvlimit = 5;
	kernelWidth = 5;

	std::vector<glm::ivec2> offset(kernelWidth * kernelWidth);

	for (int j = 0; j < kernelWidth; j++)
		for (int i = 0; i < kernelWidth; i++)
			offset.at(i + j * kernelWidth) = glm::ivec2(i - kernelWidth / 2, kernelWidth / 2 - j);

	cudaMalloc(&dev_offset, kernelWidth * kernelWidth * sizeof(glm::ivec2));
	cudaMemcpy(dev_offset, offset.data(), kernelWidth * kernelWidth * sizeof(glm::ivec2), cudaMemcpyHostToDevice);

	std::vector<float> kernel(kernelWidth * kernelWidth);
	kernel = { 1.f, 4.f, 7.f, 4.f, 1.f, 4.f, 16.f, 26.f, 16.f, 4.f, 7.f, 26.f, 41.f, 26.f, 7.f, 4.f, 16.f, 26.f, 16.f, 4.f, 1.f, 4.f, 7.f, 4.f, 1.f };
	for (int i = 0; i < kernelWidth * kernelWidth; i++)
		kernel.at(i) = kernel.at(i) / 273.f;

	cudaMalloc(&dev_kernel, kernelWidth * kernelWidth * sizeof(float));
	cudaMemcpy(dev_kernel, kernel.data(), kernelWidth * kernelWidth * sizeof(float), cudaMemcpyHostToDevice);

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
	cudaFree(dev_bufferImage);

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
		PathSegment& segment = pathSegments[index];

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
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections
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
			Geom& geom = geoms[i];

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

__global__ void shadeSimpleMaterials(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
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
		}
		else {
			segment.color = glm::vec3(0.0f);
			segment.remainingBounces = 0;
		}

		pathSegments[idx] = segment;
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
		ShadeableIntersection si = shadeableIntersections[idx];
		PathSegment ps = pathSegments[idx];
		gBuffer[ps.pixelIndex].t = si.t;
		// gBuffer[ps.pixelIndex].c += ps.color;
		gBuffer[ps.pixelIndex].c = ps.color;
		gBuffer[ps.pixelIndex].n = si.surfaceNormal;
		gBuffer[ps.pixelIndex].p = ps.ray.direction * si.t;
		//gBuffer[idx].t = si.t;
		//gBuffer[idx].c = ps.color;
		//gBuffer[idx].n = si.surfaceNormal;
		//gBuffer[idx].p = ps.ray.direction * si.t;
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, GBufferPixel* gBuffer, glm::vec3* image, PathSegment* iterationPaths)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < nPaths)
	{
		PathSegment iterationPath = iterationPaths[idx];
		gBuffer[iterationPath.pixelIndex].c = iterationPath.color;
		//image[iterationPath.pixelIndex] = gBuffer[iterationPath.pixelIndex].c;

		//PathSegment iterationPath = iterationPaths[idx];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

// Add the current iteration's output to the overall image
__global__ void denoise(glm::ivec2 resolution, 
						int stepWidth, 
						int kernelWidth,
						GBufferPixel* gBuffer,
						glm::ivec2* offset,
						float* kernel,
						glm::vec3* image)
{
	// 2-D to 1-D ***
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	//if (x < resolution.x && y < resolution.y)
	//	int index = x + (y * resolution.x);

	// 1-D to 2-D
	//int x = idx % cam.resolution.y;
	//int y = idx / cam.resolution.y;

	//int pixelCount = resolution.x * resolution.y;

	if (x < resolution.x && y < resolution.y)
	{
		int idx = x + (y * resolution.x);

		float c_phi = 1.f;
		float n_phi = 1.f;
		float p_phi = 1.f;

		glm::vec3 cval = image[idx];

		glm::vec3 sum = glm::vec3(0.f);
		GBufferPixel GBPix = gBuffer[idx];

		float cum_w = 0.f;
		for (int i = 0; i < kernelWidth; i++) {
			glm::ivec2 idx2 = (offset[i] * stepWidth) + glm::ivec2(x, y);
			idx2 = glm::clamp(idx2, glm::ivec2(0), resolution - 1);		// clamp the index of the sampling pixel

			GBufferPixel GBPix2 = gBuffer[idx2.x + (idx2.y * resolution.x)];
			glm::vec3 ctmp = image[idx2.x + (idx2.y * resolution.x)];

			glm::vec3 t = cval - ctmp;
			float dist2 = glm::dot(t, t);
			float c_w = min(exp(-dist2 / c_phi), 1.f);

			t = GBPix.n - GBPix2.n;
			dist2 = max(glm::dot(t, t) / (stepWidth * stepWidth), 0.f);
			float n_w = min(exp(-dist2 / n_phi), 1.f);

			t = GBPix.p - GBPix2.p;
			dist2 = glm::dot(t, t);
			float p_w = min(exp(-dist2 / p_phi), 1.f);

			float weight = c_w * n_w * p_w;
			sum += ctmp * weight * kernel[i];
			cum_w += weight * kernel[i];
		}

		image[idx] = sum / cum_w;
		// image[idx] = cval;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter) {

	std::cout << iter << std::endl;

	if (iter > 9)
		return;

	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
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

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
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

		if (depth == 0) {
			generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
		}

		depth++;

		shadeSimpleMaterials << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);
		iterationComplete = depth == traceDepth;
	}

	checkCUDAError("pathtrace");

	/*
	CHANGES...
	The object that will be storing the accumulated pixel values across iters is now dev_gBuffer
	instead of the dev_image. This was done so that everytime we accumulate across iters in finalGather, we are
	accumulating over the noisy values because we will be putting the final denoised image in dev_image.
	*/


	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_gBuffer, dev_image, dev_paths);

	checkCUDAError("finalGather");

	// call denoise here (2-D)
	for (int stepWidth = 1; stepWidth <= (1 << lvlimit); stepWidth = stepWidth << 1) {
		std::cout << stepWidth << std::endl;
		denoise << <blocksPerGrid2d, blockSize2d >> > (cam.resolution, stepWidth, kernelWidth, dev_gBuffer, dev_offset, dev_kernel, dev_image);
		checkCUDAError("denoise");
	}

	///////////////////////////////////////////////////////////////////////////

	// CHECKITOUT: use dev_image as reference if you want to implement saving denoised images.
	// Otherwise, screenshots are also acceptable.
	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("imageCopy");

	// this conditional check is necessary because otherwise a NULL pointer
	// will be passed to the memcpy function however that memory address
	// does not exist in host memory.
	// this occurs because data() can only be initialized
	// see line 72 of main.cpp
	if (hst_scene->state.bufferImage.data()) {
		cudaMemcpy(hst_scene->state.bufferImage.data(), dev_bufferImage,
			pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	}

	checkCUDAError("bufferImageCopy");
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
	gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer, dev_bufferImage);
}

void showImage(uchar4* pbo, int iter) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

}
