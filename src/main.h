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

#include "PerformanceTimer.h"

using namespace std;

//-------------------------------
//----------PATH TRACER----------
//-------------------------------

extern Scene* scene;
extern int iteration;

extern int width;
extern int height;

extern int ui_iterations;
extern int startupIterations;
extern bool ui_showGbuffer;
extern bool ui_denoise;
extern int ui_filterSize;
extern float ui_colorWeight;
extern float ui_normalWeight;
extern float ui_positionWeight;
extern bool ui_saveAndExit;

// Jack12 add
extern int ui_showIdx;
extern const char* ui_showItem[];
extern const int ui_ItemNum;
extern int ui_denoiseIteration;

extern glm::mat4 inverse_projection_matrix;

void runCuda(PerformanceTimer& m_timer);
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
