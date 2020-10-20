#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include "glslUtility.hpp"
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>

#include "sceneStructs.h"
#include "image.h"
#include "pathtrace.h"
#include "utilities.h"
#include "scene.h"

using namespace std;

//-------------------------------
//----------PATH TRACER----------
//-------------------------------

extern Scene* scene;
extern int iteration;

extern int width;
extern int height;

extern int ui_previewBuffer;
extern bool ui_pauseRendering;
extern int ui_limitSamples;

extern int ui_specularFilterSize;
extern float ui_specularColorWeight;
extern float ui_specularNormalWeight;
extern float ui_specularPositionWeight;

extern int ui_diffuseFilterSize;
extern float ui_diffuseColorWeight;
extern float ui_diffuseNormalWeight;
extern float ui_diffusePositionWeight;

void runCuda();
void restartRendering();
void saveImage(BufferType type);

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void scrollCallback(GLFWwindow *window, double x, double y);
