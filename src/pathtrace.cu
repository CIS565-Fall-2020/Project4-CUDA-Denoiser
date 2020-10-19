﻿#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "macros.h"

#define BLOCKSIZE2DX 8
#define BLOCKSIZE2DY 8
#define BOXSIZE 5

static float gpu_time_300_iter = 0.0f;


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
glm::vec2 signNotZero(glm::vec2 v) 
{
    return glm::vec2((v.x >= 0.0f) ? 1.0f : -1.0f, (v.y >= 0.0f) ? 1.0f : -1.0f);
}

__host__ __device__
glm::vec2 compressToOct(glm::vec3 v) 
{
    glm::vec2 p = glm::vec2(v) * (1.0f / (fabs(v.x) + fabs(v.y) + fabs(v.z)));
    return (v.z <= 0.0f) ? ((1.0f - glm::vec2(fabs(p.y), fabs(p.x))) * signNotZero(p)) : p;
}

__host__ __device__
glm::vec3 decompressToVec3(glm::vec2 e) 
{
    glm::vec3 v = glm::vec3(e.x, e.y, 1.0f - fabs(e.x) - fabs(e.y));
    if (v.z < 0) 
    {
        glm::vec2 newV = (1.0f - glm::vec2(fabs(v.y), fabs(v.x))) * signNotZero(glm::vec2(v));
        v.x = newV.x;
        v.y = newV.y;
    }
    return glm::normalize(v);
}

__host__ __device__
glm::vec3 getPosByZ(Camera& cam, float z, float curX, float curY) 
{
    float pixelDepth = z;

    glm::vec3 dir = glm::normalize(cam.view
        - cam.right * cam.pixelLength.x * ((float)curX - (float)cam.resolution.x * 0.5f)
        - cam.up * cam.pixelLength.y * ((float)curY - (float)cam.resolution.y * 0.5f)
    );

    float zLen = glm::dot(dir, cam.view);

    float t = (cam.position.z - pixelDepth) / zLen;

    return cam.position + t * dir;
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
        pbo[index].x = timeToIntersect;
        pbo[index].y = timeToIntersect;
        pbo[index].z = timeToIntersect;
    }
}

void gaussianKernel(std::vector<float>& kern, const int size, const float sigma) 
{
    int center = size / 2;
    double sum = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++) 
        {
            kern.push_back((1 / (2 * PI * sigma * sigma)) * exp(-((i - center) * (i - center) + (j - center) * (j - center)) / (2 * sigma * sigma)));
            sum += kern.at(i);
        }
    }

    for (int i = 0; i < size * size; i++)
        kern.at(i) /= sum;
}


static int materialSize = 0;
static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static Geom * dev_lights = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static PathSegment* dev_paths_cache = NULL;
static PathSegment* dev_intersections_cache = NULL;
static BoundingBox* dev_bounding_box = NULL;
static std::vector<cudaTextureObject_t> cudaTextures;
static std::vector<cudaArray*> cudaTextureData;
static glm::vec2* dev_texDim = NULL;
static cudaTextureObject_t* dev_cudaTextures = NULL;
static example::Material* dev_gltfMateiral = NULL;
static Octree* dev_octree = NULL;
static OctreeNode* dev_octreeNode = NULL;
static int* dev_primsForOcts = NULL;
static int* dev_meshTriForOcts = NULL;
static int lightLen = 0;
static GBufferPixel* dev_gBuffer = NULL;
static float* dev_aTrousKernel = NULL;
static float* dev_colLen = NULL;
static float* dev_posLen = NULL;
static float* dev_norLen = NULL;
static glm::vec3* dev_denoised_image = NULL;
static glm::vec3* dev_tmp_image = NULL;
static float* dev_gaussianKernel = NULL;
const static int depthSize = 0;

cudaTextureObject_t texTest;

// Mesh Loading
static float* dev_mesh_pos = NULL;
static float* dev_mesh_nor = NULL;
static int* dev_mesh_idx = NULL;
static float* dev_mesh_uv = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    int sampleCount = pixelcount;
    materialSize = scene->materials.size();

#ifdef ANTIALIASING
    sampleCount *= AASAMPLENUM;
#endif
    lightLen = scene->lights.size();

    

    float aTrousKernel[25] = { 1.0f / 256.0f, 1.0f / 64.0f, 3.0f / 128.0f, 1.0f / 64.0f, 1.0f / 256.0f,
                           1.0f / 64.0f, 1.0f / 16.0f, 3.0f / 32.0f, 1.0f / 16.0f, 1.0f / 64.0f,
                           3.0f / 128.0f, 3.0f / 32.0f, 9.0f / 64.0f, 3.0f / 32.0f, 3.0f / 128.0f,
                           1.0f / 64.0f, 1.0f / 16.0f, 3.0f / 32.0f, 1.0f / 16.0f, 1.0f / 64.0f,
                           1.0f / 256.0f, 1.0f / 64.0f, 3.0f / 128.0f, 1.0f / 64.0f, 1.0f / 256.0f};

    std::vector<float> hostGaussianKern;
    gaussianKernel(hostGaussianKern, 16, 1);

    cudaMalloc(&dev_gaussianKernel, 256 * sizeof(float));
    cudaMemcpy(dev_gaussianKernel, hostGaussianKern.data(), 256 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_aTrousKernel, 25 * sizeof(float));
    cudaMemcpy(dev_aTrousKernel, &(aTrousKernel[0]), 25 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_denoised_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_tmp_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_tmp_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, sampleCount * sizeof(PathSegment));
    cudaMalloc(&dev_paths_cache, sampleCount * sizeof(PathSegment));

    cudaMalloc(&dev_colLen, sampleCount * sizeof(float));
    cudaMalloc(&dev_posLen, sampleCount * sizeof(float));
    cudaMalloc(&dev_norLen, sampleCount * sizeof(float));
   
  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Geom));
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_bounding_box, scene->boundingBoxes.size() * sizeof(BoundingBox));
    cudaMemcpy(dev_bounding_box, scene->boundingBoxes.data(), scene->boundingBoxes.size() * sizeof(BoundingBox), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, sampleCount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, sampleCount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_intersections_cache, sampleCount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections_cache, 0, sampleCount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_gltfMateiral, scene->gltfMaterials.size() * sizeof(example::Material));
    cudaMemcpy(dev_gltfMateiral, scene->gltfMaterials.data(), scene->gltfMaterials.size() * sizeof(example::Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_octree, sizeof(scene->octree));
    cudaMemcpy(dev_octree, &scene->octree, sizeof(scene->octree), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_octreeNode, sizeof(OctreeNode) * scene->octree.nodeData.size());
    cudaMemcpy(dev_octreeNode, scene->octree.nodeData.data(), sizeof(OctreeNode) * scene->octree.nodeData.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_mesh_idx, scene->faceCount * 3 * sizeof(int));
    cudaMalloc(&dev_mesh_pos, scene->posCount * sizeof(float));
    cudaMalloc(&dev_mesh_nor, scene->faceCount * 3 * 3 * sizeof(float));
    cudaMalloc(&dev_mesh_uv, scene->faceCount * 2 * 3 * sizeof(float));

    int curOffset = 0;
    int curPosOffset = 0;
 
    for (int i = 0; i < scene->meshes.size(); i++) 
    {
        for (int j = 0; j < scene->meshes.at(i).size(); j++) 
        {
            int stride = scene->meshes.at(i).at(j).faces.size() / (scene->meshes.at(i).at(j).stride / sizeof(float));
            int curPosNum = scene->meshes.at(i).at(j).vertices.size();
            int final = scene->meshes.at(i).at(j).faces.at(stride * 3 - 1);
            cudaMemcpy(dev_mesh_idx + curOffset * 3,
                       scene->meshes.at(i).at(j).faces.data(), 
                       stride * 3 * sizeof(int),
                       cudaMemcpyHostToDevice);

            cudaMemcpy(dev_mesh_pos + curPosOffset,
                scene->meshes.at(i).at(j).vertices.data(),
                curPosNum * sizeof(float),
                cudaMemcpyHostToDevice);

            cudaMemcpy(dev_mesh_nor + curOffset * 3 * 3,
                scene->meshes.at(i).at(j).facevarying_normals.data(),
                stride * 3 * 3 * sizeof(float),
                cudaMemcpyHostToDevice);

            //Load UV   
            cudaMemcpy(dev_mesh_uv + curOffset * 2 * 3,
                scene->meshes.at(i).at(j).facevarying_uvs.data(),
                stride * 2 * 3 * sizeof(float),
                cudaMemcpyHostToDevice);

            curOffset += stride;
            curPosOffset += curPosNum;
        }
    }

    // Load Prim and MeshTri Oct Data
    cudaMalloc(&dev_primsForOcts, scene->octree.primitiveCount * sizeof(int));
    cudaMalloc(&dev_meshTriForOcts, scene->octree.meshTriCount * sizeof(int));

    int curPrimOffset = 0;
    int curMeshTriOffset = 0;

    for (int i = 0; i < scene->octree.nodeData.size(); i++)
    {
        cudaMemcpy(dev_primsForOcts + curPrimOffset,
            scene->octree.nodeData.at(i).primitiveIndices.data(),
            sizeof(int) * scene->octree.nodeData.at(i).primitiveCount,
            cudaMemcpyHostToDevice);

        cudaMemcpy(dev_meshTriForOcts + curMeshTriOffset,
            scene->octree.nodeData.at(i).meshTriangleIndices.data(),
            sizeof(int) * scene->octree.nodeData.at(i).meshTriCount,
            cudaMemcpyHostToDevice);

        curPrimOffset += scene->octree.nodeData.at(i).primitiveCount;
        curMeshTriOffset += scene->octree.nodeData.at(i).meshTriCount;
    }


    // Load Textures
    int count = 0;
    std::vector<glm::vec2> texDim;
    for (const auto &tex : scene->gltfTextures) 
    {
        float4* texTmp = new float4[tex.height * tex.width];

        for (int i = 0; i < tex.height * tex.width; i++)
        {
            texTmp[i].x = (float)tex.image[4 * i];
            texTmp[i].y = (float)tex.image[4 * i + 1];
            texTmp[i].z = (float)tex.image[4 * i + 2];
            texTmp[i].w = (float)tex.image[4 * i + 3];
        }

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

        // Load Data
        cudaArray* cuArray;
        cudaMallocArray(&cuArray, &channelDesc, tex.width, tex.height);

        cudaMemcpyToArray(cuArray, 0, 0, texTmp, tex.height * tex.width * sizeof(float4), cudaMemcpyHostToDevice);

        // Specify Texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        delete []texTmp;

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));

        switch (tex.sampler.wrapS) 
        {
            case CLAMP_TO_EDGE:
                texDesc.addressMode[0] = cudaAddressModeClamp;
                break;
            case MIRRORED_REPEAT:
                texDesc.addressMode[0] = cudaAddressModeMirror;
                break;
            case REPEAT:
                texDesc.addressMode[0] = cudaAddressModeWrap;
                break;
        }

        switch (tex.sampler.wrapT)
        {
            case CLAMP_TO_EDGE:
                texDesc.addressMode[1] = cudaAddressModeClamp;
                break;
            case MIRRORED_REPEAT:
                texDesc.addressMode[1] = cudaAddressModeMirror;
                break;
            case REPEAT:
                texDesc.addressMode[1] = cudaAddressModeWrap;
                break;
        }

        switch (tex.sampler.minFilter) 
        {
            case NEAREST:
            case NEAREST_MIPMAP_NEAREST:
            case NEAREST_MIPMAP_LINEAR:
                texDesc.filterMode = cudaFilterModePoint;
                break;

            case LINEAR:
            case LINEAR_MIPMAP_NEAREST:
            case LINEAR_MIPMAP_LINEAR:
                texDesc.filterMode = cudaFilterModeLinear;
                break;
        }

        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaTextureObject_t texObj = 0;
        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

        cudaTextureData.push_back(cuArray);
        cudaTextures.push_back(texObj);
        texDim.push_back(glm::vec2(tex.width, tex.height));
        count++;
    }

    cudaMalloc(&dev_texDim, texDim.size() * sizeof(glm::vec2));
    cudaMemcpy(dev_texDim, texDim.data(), texDim.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_cudaTextures, cudaTextures.size() * sizeof(cudaTextureObject_t));
    cudaMemcpy(dev_cudaTextures, cudaTextures.data(), cudaTextures.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
    cudaFree(dev_paths_cache);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
    cudaFree(dev_lights);
  	cudaFree(dev_intersections);
    cudaFree(dev_intersections_cache);
    cudaFree(dev_gltfMateiral);
    cudaFree(dev_texDim);
    cudaFree(dev_gBuffer);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_mesh_idx);
    cudaFree(dev_mesh_nor);
    cudaFree(dev_mesh_pos);
    cudaFree(dev_denoised_image);
    cudaFree(dev_tmp_image);
    cudaFree(dev_colLen);
    cudaFree(dev_norLen);
    cudaFree(dev_posLen);

    cudaFree(dev_mesh_uv);

    // Octree
    cudaFree(dev_octreeNode);
    cudaFree(dev_octree);

    for (int i = 0; i < cudaTextureData.size(); i++) 
    {
        cudaFreeArray(cudaTextureData.at(i));
        
    }
    cudaTextureData.clear();
    cudaTextures.clear();

    cudaFree(dev_cudaTextures);
    

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

    int aaDim = sqrtf(AASAMPLENUM);

    int index = x + (y * cam.resolution.x * aaDim);

    x /= aaDim;
    y /= aaDim;

	if (x < cam.resolution.x && y < cam.resolution.y) {

		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

       

		// TODO: implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);

#ifdef ANTIALIASING
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
			);
#else
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
#endif

#ifdef DEPTHOFFIELD
        // Depth of Field
        float lenX = u01(rng);
        float lenY = u01(rng);
        glm::vec2 pLens = cam.lensRadius * concentricSampling(glm::vec2(lenX, lenY));

        float ft = cam.focalDistance / segment.ray.direction.z;
        glm::vec3 pFocus = segment.ray.origin + cam.focalDistance * segment.ray.direction;

        segment.ray.origin += glm::vec3(pLens.x, pLens.y, 0.0f);
        segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
#endif

		segment.pixelIndex = x + y * cam.resolution.x;
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
    , BoundingBox* bbs
	, int geoms_size
	, ShadeableIntersection * intersections
    , float* meshPos
    , float* meshNor
    , int* meshIdx
    , float* meshUV
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
        glm::vec2 uv;
        glm::vec3 tangent;
        glm::vec3 bitangent;

		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;
        bool finalMesh = false;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;
        glm::vec3 tmp_tangent;
        glm::vec3 tmp_bitangent;

        intersections[path_index].isMesh = false;

		// naive parse through global geoms
        float isMesh = false;
		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                isMesh = false;
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                isMesh = false;
			}
            else if (geom.type == MESH) 
            {
#ifdef BOUNDINGBOX
                // Bounding Box Test
                Ray modelRay;
                modelRay.origin = multiplyMV(geom.inverseTransform, 
                                             glm::vec4(pathSegment.ray.origin, 1.0f));
                modelRay.direction = glm::normalize(multiplyMV(geom.inverseTransform, 
                                                               glm::vec4(pathSegment.ray.direction, 0.0f)));

                Geom boundingBox;
                boundingBox.type = CUBE;
                int boundingIdx = geom.boundingIdx;
                boundingBox.translation = bbs[boundingIdx].boundingCenter;
                boundingBox.scale = bbs[boundingIdx].boundingScale;
                boundingBox.rotation = glm::vec3(0.0f, 0.0f, 0.0f);

                glm::mat4 translationMat = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, 0.0f));
                glm::mat4 rotationMat = glm::rotate(glm::mat4(), boundingBox.rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
                rotationMat = rotationMat * glm::rotate(glm::mat4(), boundingBox.rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
                rotationMat = rotationMat * glm::rotate(glm::mat4(), boundingBox.rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
                glm::mat4 scaleMat = glm::scale(glm::mat4(), boundingBox.scale);
                boundingBox.transform = translationMat * rotationMat * scaleMat;

                boundingBox.inverseTransform = glm::inverse(boundingBox.transform);
                boundingBox.invTranspose = glm::inverseTranspose(boundingBox.transform);
                glm::vec3 bond_intersect = glm::vec3(0.0f);
                glm::vec3 bond_normal = glm::vec3(0.0f);
                bool bond_outside = true;

                t = boxIntersectionTest(boundingBox, modelRay, bond_intersect, bond_normal, bond_outside);

                /*t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside,
                    meshPos, meshNor, meshIdx, geom.faceNum, geom.offset);*/

                if (t != -1) 
                {
                    t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside, tmp_tangent, tmp_bitangent,
                        meshPos, meshNor, meshIdx, meshUV, geom.faceNum, geom.offset, geom.posOffset);
                } 
#else
                t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside, tmp_tangent, tmp_bitangent,
                    meshPos, meshNor, meshIdx, meshUV, geom.faceNum, geom.offset, geom.posOffset);
#endif
                isMesh = true;
            }
			
            

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
                if (isMesh) 
                {
                    finalMesh = true;
                    uv = tmp_uv;
                    tangent = tmp_tangent;
                    bitangent = tmp_bitangent;
                }
                else 
                {
                    finalMesh = false;
                }
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
            intersections[path_index].uv = uv;
            intersections[path_index].isMesh = finalMesh;
            intersections[path_index].surfaceTangent = tangent;
            intersections[path_index].surfaceBiTangent = bitangent;
		}
	}
}

__global__ void rayOctreeIntersect(
    int depth
    , int num_paths
    , PathSegment* pathSegments
    , Geom* geoms
    , BoundingBox* bbs
    , int geoms_size
    , ShadeableIntersection* intersections
    , float* meshPos
    , float* meshNor
    , int* meshIdx
    , float* meshUV
    , Octree* octTree
    , OctreeNode* octreeNode
    , int* primOct
    , int* meshTriOct)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    int octStack[OCT_MAX_DEPTH + 1];
    int childStack[OCT_MAX_DEPTH + 1];

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        intersections[path_index].isMesh = false;
        intersections[path_index].t = FLT_MAX;

        int top = 0;
        octStack[top] = 0;
        childStack[top] = 0;
        OctreeNode* curNode;

        do
        {
            int curNodeIndex = octStack[top];
            OctreeNode* curNode = &(octreeNode[curNodeIndex]);
            
            //First Coming
            if (childStack[top] == 0) 
            {
                glm::vec3 bond_intersect = glm::vec3(0.0f);
                glm::vec3 bond_normal = glm::vec3(0.0f);
                bool bond_outside = true;

                int octBoxT = boxIntersectionTest(curNode->octBlock, pathSegment.ray, bond_intersect, bond_normal, bond_outside);


                if (octBoxT != -1)
                {
                    
                    float geomT;
                    glm::vec3 intersect_point;
                    glm::vec3 normal;
                    glm::vec2 uv;
                    glm::vec3 tangent;
                    glm::vec3 bitangent;

                    float t_min = FLT_MAX;
                    int hit_geom_index = -1;
                    bool outside = true;
                    bool finalMesh = false;

                    glm::vec3 tmp_intersect;
                    glm::vec3 tmp_normal;
                    glm::vec2 tmp_uv;
                    glm::vec3 tmp_tangent;
                    glm::vec3 tmp_bitangent;

                    intersections[path_index].isMesh = false;

                    // naive parse through global geoms
                    float isMesh = false;

                    for (int i = 0; i < curNode->primitiveCount; i++) 
                    {
                        int geomIdx = primOct[curNode->primOffset + i];
                        Geom& geom = geoms[geomIdx];

                        if (geom.type == CUBE)
                        {
                            geomT = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                            isMesh = false;
                        }
                        else if (geom.type == SPHERE)
                        {
                            geomT = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                            isMesh = false;
                        }
                        else if (geom.type == MESH)
                        {

                            geomT = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside, tmp_tangent, tmp_bitangent,
                                meshPos, meshNor, meshIdx, meshUV, geom.faceNum, geom.offset, geom.posOffset);

                            isMesh = true;
                        }

                        // Compute the minimum t from the intersection tests to determine what
                        // scene geometry object was hit first.
                        if (geomT > 0.0f && t_min >= geomT)
                        {
                            if (isMesh)
                            {
                                finalMesh = true;
                                uv = tmp_uv;
                                tangent = tmp_tangent;
                                bitangent = tmp_bitangent;
                            }
                            else
                            {
                                finalMesh = false;
                            }
                            t_min = geomT;
                            hit_geom_index = geomIdx;
                            intersect_point = tmp_intersect;
                            normal = tmp_normal;
                            glm::vec3 pMin = curNode->boxCenter - glm::vec3(curNode->scale / 2.0f);
                            glm::vec3 pMax = curNode->boxCenter + glm::vec3(curNode->scale / 2.0f);
                            if (!(pMin.x < tmp_intersect.x && pMax.x > tmp_intersect.x
                                && pMin.y < tmp_intersect.y && pMax.y > tmp_intersect.y
                                && pMin.z < tmp_intersect.z && pMax.z > tmp_intersect.z)) 
                            {
                                hit_geom_index = -1;
                            }
                        }

                        
                    }

                    if (hit_geom_index != -1 && intersections[path_index].t >= t_min)
                    {
                        //The ray hits something
                        intersections[path_index].t = t_min;
                        intersections[path_index].materialId = geoms[hit_geom_index].materialid;
                        intersections[path_index].surfaceNormal = normal;
                        intersections[path_index].uv = uv;
                        intersections[path_index].isMesh = finalMesh;
                    }

                    if (curNode->childCount == 0)
                        top--;
                    else 
                    {
                        for (int i = childStack[top]; i < 8; i++)
                        {
                            if (curNode->hasChild[i])
                            {
                                childStack[top] = i + 1;
                                top++;
                                octStack[top] = curNode->nodeIndices[i];
                                childStack[top] = 0;

                                break;
                            }
                        }
                    }
                }
                else
                {
                    top--;
                }
            }
            else 
            {
                if (childStack[top] == 8) 
                {
                    top--;
                }
                else 
                {
                    for (int i = childStack[top]; i < 8; i++)
                    {
                        if (curNode->hasChild[i])
                        {
                            childStack[top] = i + 1;
                            top++;
                            octStack[top] = curNode->nodeIndices[i];
                            childStack[top] = 0;

                            break;
                        }
                        else if (i == 7) 
                        {
                            childStack[top] = 8;
                        }
                    }
                }
                
            }
        } while (top >= 0);

        if (intersections[path_index].t == FLT_MAX) 
        {
            intersections[path_index].t = -1.0f;
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
__global__ void shadeMaterial(
    int iter
    , int num_paths
    , ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials
    , cudaTextureObject_t* cudaTexes
    , int materialSize
    , example::Material* gltfMaterials
    , glm::vec2* texDim
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

      Material material;
      bool isMesh = false;
      if (intersection.materialId > materialSize - 1)
      {
          isMesh = true;
          example::Material curMeshMaterial = gltfMaterials[intersection.materialId - materialSize];
          int baseColorIndex = curMeshMaterial.base_texid;
          int normalIndex = curMeshMaterial.normal_texid;
          int metallicIndex = curMeshMaterial.metallic_roughness_texid;

          if (baseColorIndex != -1)
          {
              float width = texDim[baseColorIndex].x;
              float height = texDim[baseColorIndex].y;
              float4 color = tex2D<float4>(cudaTexes[baseColorIndex], (intersection.uv.x) * width, (intersection.uv.y) * height);
              material.color = glm::vec3(color.x / 255.0f, color.y / 255.0f, color.z / 255.0f);
              material.specular.color = glm::vec3(color.x / 255.0f, color.y / 255.0f, color.z / 255.0f);
          }
          else
          {
              material.color = glm::vec3(0.98f, 0.98f, 0.98f);
          }

          if (normalIndex != -1)
          {
              float width = texDim[normalIndex].x;
              float height = texDim[normalIndex].y;

              float4 normal = tex2D<float4>(cudaTexes[normalIndex], (intersection.uv.x) * width, (intersection.uv.y) * height);

              intersection.surfaceNormal = glm::vec3((normal.x - 128.0f) / 128.0f,
                  (normal.y - 128.0f) / 128.0f,
                  (normal.z - 128.0f) / 128.0f);
          }

          if (metallicIndex != -1)
          {
              float width = texDim[normalIndex].x;
              float height = texDim[normalIndex].y;

              float4 metallic = tex2D<float4>(cudaTexes[metallicIndex], (intersection.uv.x) * width, (intersection.uv.y) * height);
              material.hasReflective = metallic.y / 255.0f;

          }

          material.hasRefractive = 0;
          material.emittance = 0;
      }
      else
      {
          isMesh = false;
          material = materials[intersection.materialId];
      }
     
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
        pathSegments[idx].remainingBounces = 0;
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
          scatterRay(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, material, rng, isMesh);
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
      pathSegments[idx].remainingBounces = 0;
    }
  }
}

__global__ void directLightShadeMaterial(
    int iter
    , int num_paths
    , ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials
    , Geom* lights
    , int lightNum
    , cudaTextureObject_t* cudaTexes
    , int materialSize
    , example::Material* gltfMaterials
    , glm::vec2* texDim
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

            Material material;
            if (intersection.materialId > materialSize - 1)
            {
                example::Material curMeshMaterial = gltfMaterials[intersection.materialId - materialSize];
                int baseColorIndex = curMeshMaterial.base_texid;
                int normalIndex = curMeshMaterial.normal_texid;
                int metallicIndex = curMeshMaterial.metallic_roughness_texid;

                if (baseColorIndex != -1) 
                {
                    float width = texDim[baseColorIndex].x;
                    float height = texDim[baseColorIndex].y;
                    float4 color = tex2D<float4>(cudaTexes[baseColorIndex], (intersection.uv.x) * width, (intersection.uv.y) * height);
                    material.color = glm::vec3(color.x / 255.0f, color.y / 255.0f, color.z / 255.0f);
                }
                else 
                {
                    material.color = glm::vec3(0.98f, 0.98f, 0.98f);
                }
                
                if (normalIndex != -1)
                {
                    float width = texDim[normalIndex].x;
                    float height = texDim[normalIndex].y;

                    float4 normal = tex2D<float4>(cudaTexes[normalIndex], (intersection.uv.x) * width, (intersection.uv.y) * height);
                   
                    intersection.surfaceNormal = glm::vec3((normal.x - 128.0f) / 128.0f,
                        (normal.y - 128.0f) / 128.0f,
                        (normal.z - 128.0f) / 128.0f);
                }

                if (metallicIndex != -1)
                {
                    float width = texDim[normalIndex].x;
                    float height = texDim[normalIndex].y;

                    float4 metallic = tex2D<float4>(cudaTexes[metallicIndex], (intersection.uv.x) * width, (intersection.uv.y) * height);
                    //material.hasReflective = metallic.x / 255.0f;

                }
                
                material.hasReflective = 0;
                material.hasRefractive = 0;
                material.emittance = 0;
            }
            else
            {
                material = materials[intersection.materialId];
            }

            //material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                if (pathSegments[idx].remainingBounces != 1) 
                {
                    pathSegments[idx].color *= (materialColor * material.emittance);
                }
                else 
                {
                    pathSegments[idx].color *= (materialColor * material.emittance) 
                                               / glm::length2(getPointOnRay(pathSegments[idx].ray, intersection.t) - pathSegments[idx].ray.origin)
                                               * fabs(glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction));
                }
                
                pathSegments[idx].remainingBounces = 0;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                if (pathSegments[idx].remainingBounces == 1) 
                {
                    pathSegments[idx].color = glm::vec3(0.0f);
                    pathSegments[idx].remainingBounces = 0;
                }
                else 
                {
                    pathSegments[idx].color = glm::vec3(0.0f);
                    directRay(pathSegments[idx],
                        getPointOnRay(pathSegments[idx].ray, intersection.t),
                        intersection.surfaceNormal, 
                        material, rng, lights, lightNum);
                }
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
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
        gBuffer[idx].t = shadeableIntersections[idx].t;
#ifdef GBUFFEROPT
        gBuffer[idx].z = getPointOnRay(pathSegments[idx].ray, shadeableIntersections[idx].t).z;
        gBuffer[idx].octNor = compressToOct(shadeableIntersections[idx].surfaceNormal);
#else
        gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
        gBuffer[idx].pos = getPointOnRay(pathSegments[idx].ray, shadeableIntersections[idx].t);
#endif
        
    }
}

__global__ void getLength(GBufferPixel* gBuffer, glm::vec3* image, float* colLen, float* posLen, float* norLen, int nPaths, Camera cam)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths) 
    {
        int pixelY = index / cam.resolution.x;
        int pixelX = index - pixelY * cam.resolution.x;

        colLen[index] = glm::length(image[index]);
#ifdef GBUFFEROPT
        posLen[index] = glm::length(getPosByZ(cam, gBuffer[index].z, pixelX, pixelY));
        norLen[index] = glm::length(decompressToVec3(gBuffer[index].octNor));
#else
        posLen[index] = glm::length(gBuffer[index].pos);
        norLen[index] = glm::length(gBuffer[index].normal);
#endif
    }
}

__global__ void getVariances(float* colLen, float* posLen, float* norLen, int nPaths, float colAvg, float norAvg, float posAvg) 
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        colLen[index] = (colLen[index] - colAvg) * (colLen[index] - colAvg);
        posLen[index] = (posLen[index] - posAvg) * (posLen[index] - posAvg);
        norLen[index] = (norLen[index] - norAvg) * (norLen[index] - norAvg);
    }
}

__global__ void aTrousFilter(GBufferPixel* gBuffer, glm::vec3* image, glm::vec3* denoised_image, int iter, int width, int height, float* filter,
                             float cPhi, float nPhi, float pPhi, int nPaths, Camera cam) 
{

    int pixelY = blockDim.x * blockIdx.x + threadIdx.x;
    int pixelX = blockDim.y * blockIdx.y + threadIdx.y;

    

    int index = pixelY * width + pixelX;

    if (index < nPaths) 
    {
        glm::vec3 pixelCol = image[index];
        
#ifdef GBUFFEROPT
        glm::vec3 pixelPos = getPosByZ(cam, gBuffer[index].z, pixelX, pixelY);
        glm::vec3 pixelNor = decompressToVec3(gBuffer[index].octNor);
#else
        glm::vec3 pixelPos = gBuffer[index].pos;
        glm::vec3 pixelNor = gBuffer[index].normal;
#endif
        int step = 1 << iter;
        
#ifdef SHAREDFILTER
       
        __shared__ GBufferPixel smem[BLOCKSIZE2DX + BOXSIZE - 1][BLOCKSIZE2DY + BOXSIZE - 1];
        
        if (step == 1) 
        {
            // Load Shared Memory
            if (threadIdx.x < BOXSIZE - 1 && threadIdx.y < BOXSIZE - 1)
            {
                int upperLeftX = pixelX - 2 * step;
                int upperLeftY = pixelY - 2 * step;

                int upperLeftIdx = upperLeftY * width + upperLeftX;

                if (upperLeftIdx >= 0 && upperLeftIdx < nPaths)
                    smem[threadIdx.x][threadIdx.y] = gBuffer[upperLeftIdx];
            }

            if (threadIdx.x < BOXSIZE - 1)
            {
                int upperRightX = pixelX + 2 * step;
                int upperRightY = pixelY - 2 * step;

                int upperRightIdx = upperRightY * width + upperRightX;

                if (upperRightIdx >= 0 && upperRightIdx < nPaths)
                    smem[threadIdx.x + BOXSIZE - 1][threadIdx.y] = gBuffer[upperRightIdx];
            }


            if (threadIdx.y < BOXSIZE - 1)
            {
                int lowerLeftX = pixelX - 2 * step;
                int lowerLeftY = pixelY + 2 * step;

                int lowerLeftIdx = lowerLeftY * width + lowerLeftX;

                if (lowerLeftIdx >= 0 && lowerLeftIdx < nPaths)
                    smem[threadIdx.x][threadIdx.y + BOXSIZE - 1] = gBuffer[lowerLeftIdx];
            }

            int lowerRightX = pixelX + 2 * step;
            int lowerRightY = pixelY + 2 * step;

            int lowerRightIdx = lowerRightY * width + lowerRightX;
            if (lowerRightIdx >= 0 && lowerRightIdx < nPaths)
                smem[threadIdx.x + BOXSIZE - 1][threadIdx.y + BOXSIZE - 1] = gBuffer[lowerRightIdx];

            __syncthreads();
        }

#endif
        
        glm::vec3 sum = glm::vec3(0.0f);
        float cum_w = 0.0f;
        for (int i = -2; i <= 2; i++)
        {
            for (int j = -2; j <= 2; j++)
            {
#ifdef SHAREDFILTER
                GBufferPixel curPixel;
                int curX = pixelX + j * step;
                int curY = pixelY + i * step;

                curX = glm::clamp(curX, 0, width - 1);
                curY = glm::clamp(curY, 0, height - 1);

                int curIdx = curY * width + curX;
                if (curIdx < 0 || curIdx >= nPaths)
                    continue;
                if (step == 1) 
                    curPixel = smem[threadIdx.x + j + 2][threadIdx.y + i + 2];
                else 
                    curPixel = gBuffer[curIdx];
#else
                int curX = pixelX + j * step;
                int curY = pixelY + i * step;

                curX = glm::clamp(curX, 0, width - 1);
                curY = glm::clamp(curY, 0, height - 1);

                int curIdx = curY * width + curX;

                if (curIdx < 0 || curIdx >= nPaths)
                    continue;

                GBufferPixel curPixel = gBuffer[curIdx];
#endif

#ifdef GBUFFEROPT
                glm::vec3 curPos = getPosByZ(cam, curPixel.z, curX, curY);
                glm::vec3 curNor = decompressToVec3(curPixel.octNor);
#else
                glm::vec3 curNor = curPixel.normal;
                glm::vec3 curPos = curPixel.pos;
#endif
                glm::vec3 curCol = image[curIdx];

                // Color Weight
                glm::vec3 t = pixelCol - curCol;
                float dist2 = glm::dot(t, t);
                float c_w = min(expf(-(dist2) / cPhi), 1.0f);

                // Normal Weight
                t = pixelNor - curNor;
                dist2 = max(glm::dot(t, t) / (step * step), 0.0f);
                float n_w = min(expf(-(dist2) / nPhi), 1.0f);

                // Pos Weight
                t = pixelPos - curPos;
                dist2 = dot(t, t);
                float p_w = min(expf(-(dist2) / pPhi), 1.0f);

                float weight = c_w * n_w * p_w;
                sum += curCol * weight * filter[(i + 2) * 5 + (j + 2)];
                cum_w += weight * filter[(i + 2) * 5 + (j + 2)];
            }
        }
        denoised_image[index] = sum / cum_w;  
    }
}

__global__ void gaussianFilter(GBufferPixel* gBuffer, glm::vec3* image, glm::vec3* denoised_image, int width, int height, float* filter,
    float cPhi, float nPhi, float pPhi, int nPaths, Camera cam)
{

    int pixelY = blockDim.x * blockIdx.x + threadIdx.x;
    int pixelX = blockDim.y * blockIdx.y + threadIdx.y;

    int index = pixelY * width + pixelX;


    if (index < nPaths)
    {
        glm::vec3 pixelCol = image[index];

#ifdef GBUFFEROPT
        glm::vec3 pixelPos = getPosByZ(cam, gBuffer[index].z, pixelX, pixelY);
        glm::vec3 pixelNor = decompressToVec3(gBuffer[index].octNor);
#else
        glm::vec3 pixelPos = gBuffer[index].pos;
        glm::vec3 pixelNor = gBuffer[index].normal;
#endif

        glm::vec3 sum = glm::vec3(0.0f);
        float cum_w = 0.0f;
        for (int i = -7; i <= 8; i++)
        {
            for (int j = -7; j <= 8; j++)
            {
                int curX = pixelX + j;
                int curY = pixelY + i;

                curX = glm::clamp(curX, 0, width - 1);
                curY = glm::clamp(curY, 0, height - 1);

                int curIdx = curY * width + curX;
                GBufferPixel curPixel = gBuffer[curIdx];
#ifdef GBUFFEROPT
                glm::vec3 curPos = getPosByZ(cam, curPixel.z, curX, curY);
                glm::vec3 curNor = decompressToVec3(curPixel.octNor);
#else
                glm::vec3 curNor = gBuffer[curIdx].normal;
                glm::vec3 curPos = gBuffer[curIdx].pos;
#endif
                glm::vec3 curCol = image[curIdx];

                // Color Weight
                glm::vec3 t = pixelCol - curCol;
                float dist2 = glm::dot(t, t);
                float c_w = min(expf(-(dist2) / cPhi), 1.0f);

                // Normal Weight
                t = pixelNor - curNor;
                dist2 = max(glm::dot(t, t), 0.0f);
                float n_w = min(expf(-(dist2) / nPhi), 1.0f);

                // Pos Weight
                t = pixelPos - curPos;
                dist2 = dot(t, t);
                float p_w = min(expf(-(dist2) / pPhi), 1.0f);

                float weight = c_w * n_w * p_w;
                sum += curCol * weight * filter[(i + 7) * 16 + (j + 7)];
                cum_w += weight * filter[(i + 7) * 16 + (j + 7)];
            }
        }
        denoised_image[index] = sum / cum_w;
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
#ifdef ANTIALIASING
        iterationPath.color /= AASAMPLENUM;
#endif
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter, float cPhi, float nPhi, float pPhi, int filterIter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    int sampleCount = pixelcount;

#ifdef ANTIALIASING
    sampleCount *= AASAMPLENUM;
#endif

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);

#ifdef ANTIALIASING
    const dim3 blocksPerGrid2d(
        (cam.resolution.x * sqrt(AASAMPLENUM) + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y * sqrt(AASAMPLENUM)+ blockSize2d.y - 1) / blockSize2d.y);
#else
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
#endif
    

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

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_end = nullptr;

    cudaEventCreate(&event_start);
    cudaEventCreate(&event_end);

    int depth = 0;

    PathSegment* dev_path_end = dev_paths + sampleCount; //the tail of path segment array
    int num_paths = dev_path_end - dev_paths; //is that the same as pixel count? -- no when antialiasing, do we need to change?
    int num_cur_paths = num_paths;
    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    bool firstIteration = true;

    cudaEventRecord(event_start);

    while (!iterationComplete) {

        // clean shading chunks
        cudaMemset(dev_intersections, 0, sampleCount * sizeof(ShadeableIntersection));
        dim3 numblocksPathSegmentTracing = (num_cur_paths + blockSize1d - 1) / blockSize1d;
        if (CACHEBOUNCE && depth == 0)
        {
            if (iter == 1)
            {
#ifdef OCTREEACCEL
                rayOctreeIntersect << <numblocksPathSegmentTracing, blockSize1d >> >(
                    depth
                    , num_paths
                    , dev_paths
                    , dev_geoms
                    , dev_bounding_box
                    , hst_scene->geoms.size()
                    , dev_intersections
                    , dev_mesh_pos
                    , dev_mesh_nor
                    , dev_mesh_idx
                    , dev_mesh_uv
                    , dev_octree
                    , dev_octreeNode
                    , dev_primsForOcts
                    , dev_meshTriForOcts);
#else
                // tracing
                computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                    depth
                    , num_paths
                    , dev_paths
                    , dev_geoms
                    , dev_bounding_box
                    , hst_scene->geoms.size()
                    , dev_intersections
                    , dev_mesh_pos
                    , dev_mesh_nor
                    , dev_mesh_idx
                    , dev_mesh_uv
                    );
#endif
                checkCUDAError("trace one bounce");
                cudaDeviceSynchronize();

                cudaMemcpy(dev_paths_cache, dev_paths, sampleCount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
                cudaMemcpy(dev_intersections_cache, dev_intersections, sampleCount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            }
            else 
            {
                cudaMemcpy(dev_paths, dev_paths_cache, sampleCount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
                cudaMemcpy(dev_intersections, dev_intersections_cache, sampleCount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            }         
        }
        else 
        {
#ifdef OCTREEACCEL
            rayOctreeIntersect << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , dev_bounding_box
                , hst_scene->geoms.size()
                , dev_intersections
                , dev_mesh_pos
                , dev_mesh_nor
                , dev_mesh_idx
                , dev_mesh_uv
                , dev_octree
                , dev_octreeNode
                , dev_primsForOcts
                , dev_meshTriForOcts);
#else
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , dev_bounding_box
                , hst_scene->geoms.size()
                , dev_intersections
                , dev_mesh_pos
                , dev_mesh_nor
                , dev_mesh_idx
                , dev_mesh_uv
                );
#endif
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
        }

        if (depth == 0) {
            generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
        }

        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
          // evaluating the BSDF.
          // Start off with just a big kernel that handles all the different
          // materials you have in the scenefile.

          // TODO: compare between directly shading the path segments and shading
          // path segments that have been reshuffled to be contiguous in memory.
        
        // Sort Path with Matrial ID
        thrust::stable_sort_by_key(thrust::device, dev_intersections, dev_intersections + num_cur_paths, dev_paths, material_sort());
#ifdef DIRECTLIGHTING
        directLightShadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_lights,
            lightLen,
            dev_cudaTextures,
            materialSize,
            dev_gltfMateiral,
            dev_texDim
            );
#else
        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_cudaTextures,
            materialSize,
            dev_gltfMateiral,
            dev_texDim
            );
#endif
        
        cudaDeviceSynchronize();
        // TODO: should be based off stream compaction results, and even shot more rays
        // update the dev_path and num_paths
       
        dev_path_end = thrust::partition(
            thrust::device_ptr<PathSegment>(dev_paths),
            thrust::device_ptr<PathSegment>(dev_path_end),
            is_terminated()).get();

        //dev_path_end = thrust::remove_if(thrust::device, dev_paths, dev_path_end, is_terminated());

        num_cur_paths = dev_path_end - dev_paths;

        if ((depth >= traceDepth) || num_cur_paths == 0)
            iterationComplete = true;
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (sampleCount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);
    //thrust::stable_sort_by_key(thrust::device, dev_paths, dev_paths + sampleCount, dev_gBuffer, pixel_sort());
    getLength << <numBlocksPixels, blockSize1d >> > (dev_gBuffer, dev_image, dev_colLen, dev_posLen, dev_norLen, sampleCount, cam);
    float colAvg = thrust::reduce(thrust::device, dev_colLen, dev_colLen + sampleCount) / (float)sampleCount;
    float norAvg = thrust::reduce(thrust::device, dev_norLen, dev_norLen + sampleCount) / (float)sampleCount;
    float posAvg = thrust::reduce(thrust::device, dev_posLen, dev_posLen + sampleCount) / (float)sampleCount;

    getVariances << <numBlocksPixels, blockSize1d >> > (dev_colLen, dev_posLen, dev_norLen, sampleCount, colAvg, norAvg, posAvg);
    checkCUDAError("pathtrace");

    float colVar = thrust::reduce(thrust::device, dev_colLen, dev_colLen + sampleCount) / (float)sampleCount;
    float norVar = thrust::reduce(thrust::device, dev_norLen, dev_norLen + sampleCount) / (float)sampleCount;
    float posVar = thrust::reduce(thrust::device, dev_posLen, dev_posLen + sampleCount) / (float)sampleCount;

    cudaMemcpy(dev_tmp_image, dev_image, sizeof(glm::vec3)* sampleCount, cudaMemcpyDeviceToDevice);

#ifdef GAUSSIAN
    gaussianFilter << <blocksPerGrid2d, blockSize2d >> > (dev_gBuffer, dev_tmp_image, dev_denoised_image, cam.resolution.x, cam.resolution.y,
        dev_gaussianKernel, colVar, norVar, posVar, sampleCount, cam);
#else
    for (int i = 0; i < filterIter; i++) 
    {
        aTrousFilter << <blocksPerGrid2d, blockSize2d >> > (dev_gBuffer, dev_tmp_image, dev_denoised_image, i, cam.resolution.x, cam.resolution.y,
            dev_aTrousKernel, colVar / (1 << i), norVar/ (1 << i), posVar / (1 << i), sampleCount, cam);
        cudaMemcpy(dev_tmp_image, dev_denoised_image, sampleCount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    }
#endif
    

    checkCUDAError("pathtrace");
    ///////////////////////////////////////////////////////////////////////////

    cudaEventRecord(event_end);
    cudaEventSynchronize(event_end);

    float curIterTime = 0.0f;
    
    cudaEventElapsedTime(&curIterTime, event_start, event_end);
    gpu_time_300_iter += curIterTime;
    

    if (iter == 100)
    {
        std::cout << "100 Iter Elapse Time: " << gpu_time_300_iter << "ms";
    }
    
    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
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
}

void showDenoisedImage(uchar4* pbo, int iter) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_denoised_image);
}