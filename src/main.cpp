#include <cstring>
#include <chrono>

#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_glfw.h"
#include "../imgui/imgui_impl_opengl3.h"

#include "main.h"
#include "preview.h"
#include "pathtrace.h"


constexpr std::size_t log2SqrtNumStratifiedSamples = 5;
constexpr std::size_t sqrtNumStratifiedSamples = 1 << log2SqrtNumStratifiedSamples;
constexpr std::size_t numStratifiedSamples = sqrtNumStratifiedSamples * sqrtNumStratifiedSamples;

constexpr bool enableMultipleImportanceSampling = true;


static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

int ui_previewBuffer = static_cast<int>(BufferType::AccumulatedColor);
extern bool ui_pauseRendering = false;
extern int ui_limitSamples = 0;

int ui_filterSize = 80;
float ui_colorWeight = 0.45f;
float ui_normalWeight = 0.35f;
float ui_positionWeight = 0.2f;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

std::chrono::high_resolution_clock::time_point startTime;

Scene *scene;
RenderState *renderState;
int iteration;

std::default_random_engine randGen;
std::vector<int> directLightingSamples;

int width;
int height;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
    startTimeString = currentTimeString();

    if (argc < 2) {
        printf("Usage: %s SCENEFILE.txt\n", argv[0]);
        return 1;
    }

    const char *sceneFile = argv[1];

    // Load scene file
    scene = new Scene(sceneFile);

    scene->buildTree();
    std::vector<std::pair<int, int>> stk;
    stk.emplace_back(scene->aabbTreeRoot, 0);
    int max = 0;
    while (!stk.empty()) {
        int node = stk.back().first;
        int depth = stk.back().second;
        stk.pop_back();

        max = std::max(max, depth);

        if (scene->aabbTree[node].leftChild >= 0) {
            stk.emplace_back(scene->aabbTree[node].leftChild, depth + 1);
        }
        if (scene->aabbTree[node].rightChild >= 0) {
            stk.emplace_back(scene->aabbTree[node].rightChild, depth + 1);
        }
    }
    std::cout << "Total primitives: " << scene->geoms.size() << "\n";
    std::cout << "AABB tree depth: " << max << "\n";

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    phi = glm::atan(view.x, view.z);
    theta = glm::pi<float>() - glm::acos(view.y);
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    // Initialize CUDA and GL components
    init();

    // GLFW main loop
    mainLoop();

    return 0;
}

void saveImage(BufferType type) {
    saveBufferState(type, iteration);

    float samples = iteration;
    // output image file
    image img(width, height);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            img.setPixel(width - 1 - x, y, renderState->image[index]);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp" << ".";
    switch (static_cast<BufferType>(ui_previewBuffer)) {
    case BufferType::AccumulatedColor:
        ss << "raw";
        break;
    case BufferType::Normal:
        ss << "norm";
        break;
    case BufferType::Position:
        ss << "pos";
        break;
    case BufferType::FilteredColor:
        ss << "denoised";
        break;
    }
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
    if (camchanged) {
        iteration = 0;
        Camera &cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 r = glm::normalize(glm::cross(v, glm::vec3(0, 1, 0)));
        cam.up = glm::cross(r, v);
        cam.right = r;

        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
      }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0) {
        pathtraceFree();
        pathtraceInit(scene, sqrtNumStratifiedSamples);
        /*randGen = std::default_random_engine(std::random_device{}());*/
        randGen = std::default_random_engine(1998);
        startTime = std::chrono::high_resolution_clock::now();
    }

    // always render the first iteration
    bool keepRendering = true;
    if (iteration != 0) {
        if (ui_pauseRendering) {
            keepRendering = false;
        }
        if (ui_limitSamples != 0 && iteration >= ui_limitSamples) {
            keepRendering = false;
        }
    }

    if (keepRendering) {
        if (iteration < renderState->iterations) {
            if (iteration % numStratifiedSamples == 0) {
                std::vector<std::vector<IntersectionSample>> isectVecs;
                for (int i = 0; i < renderState->traceDepth; ++i) {
                    std::vector<glm::vec2>
                        out = generateStratifiedSamples2D(sqrtNumStratifiedSamples, randGen),
                        mis1 = generateStratifiedSamples2D(sqrtNumStratifiedSamples, randGen),
                        mis2 = generateStratifiedSamples2D(sqrtNumStratifiedSamples, randGen);
                    std::vector<IntersectionSample> layer(numStratifiedSamples);
                    for (std::size_t i = 0; i < numStratifiedSamples; ++i) {
                        layer[i].out = out[i];
                        layer[i].mis1 = mis1[i];
                        layer[i].mis2 = mis2[i];
                    }
                    isectVecs.emplace_back(std::move(layer));
                }

                std::vector<glm::vec2>
                    pixel = generateStratifiedSamples2D(sqrtNumStratifiedSamples, randGen),
                    dof = generateStratifiedSamples2D(sqrtNumStratifiedSamples, randGen);
                std::vector<CameraSample> cameraVec(numStratifiedSamples);
                for (std::size_t i = 0; i < numStratifiedSamples; ++i) {
                    cameraVec[i].pixel = pixel[i];
                    cameraVec[i].dof = dof[i];
                }

                updateStratifiedSamples(isectVecs, cameraVec);
            }
            if (!scene->lightPoolMis.empty() && iteration % scene->lightPoolMis.size() == 0) {
                directLightingSamples = scene->lightPoolMis;
                std::shuffle(directLightingSamples.begin(), directLightingSamples.end(), randGen);
            }
            iteration++;

            // execute the kernel
            int frame = 0;
            int misLight = -1;
            if (enableMultipleImportanceSampling) {
                misLight =
                    directLightingSamples.empty() ? -1 : directLightingSamples[iteration % directLightingSamples.size()];
            }
            pathtrace(frame, iteration, misLight, scene->lightPoolMis.size());

            if (iteration == 1000) {
                std::cout << "1000 iterations, time: " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count() << "\n";
            }
        } else {
            saveImage(static_cast<BufferType>(ui_previewBuffer));
            pathtraceFree();
            cudaDeviceReset();
            exit(EXIT_SUCCESS);
        }
    }

    aTrous(5, ui_filterSize, iteration, ui_colorWeight, ui_normalWeight, ui_positionWeight);

    uchar4 *pbo_dptr = NULL;
    cudaGLMapBufferObject((void **)&pbo_dptr, pbo);
    sendBufferToPbo(pbo_dptr, static_cast<BufferType>(ui_previewBuffer), iteration);
    // unmap buffer object
    cudaGLUnmapBufferObject(pbo);
}

void restartRendering() {
    iteration = 0;
}


void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        switch (key) {
        case GLFW_KEY_ESCAPE:
            saveImage(static_cast<BufferType>(ui_previewBuffer));
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        case GLFW_KEY_S:
            saveImage(static_cast<BufferType>(ui_previewBuffer));
            break;
        case GLFW_KEY_SPACE:
            camchanged = true;
            renderState = &scene->state;
            Camera &cam = renderState->camera;
            cam.lookAt = ogLookAt;
            break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (ImGui::GetIO().WantCaptureMouse) {
        return;
    }
    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
    if (leftMousePressed) {
        // compute new camera parameters
        phi -= (xpos - lastX) / width;
        theta -= (ypos - lastY) / height;
        theta = std::fmax(0.001f, std::fmin(theta, PI));
        camchanged = true;
    } else if (rightMousePressed) {
        Camera &cam = renderState->camera;
        cam.fovy += 10.0f * (ypos - lastY) / height;
        cam.fovy = std::max(cam.fovy, 5.0f);
        Scene::computeCameraParameters(cam);
        camchanged = true;
    } else if (middleMousePressed) {
        renderState = &scene->state;
        Camera &cam = renderState->camera;
        glm::vec3 up = cam.up;
        glm::vec3 right = cam.right;
        right.y = 0.0f;
        right = glm::normalize(right);

        cam.lookAt -= (float) (xpos - lastX) * right * 0.01f;
        cam.lookAt += (float) (ypos - lastY) * up * 0.01f;
        camchanged = true;
    }
    lastX = xpos;
    lastY = ypos;
}

void scrollCallback(GLFWwindow *window, double x, double y) {
    zoom -= y * 0.1f;
    zoom = std::fmax(0.1f, zoom);
    camchanged = true;
}
