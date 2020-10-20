#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void filterInit(int filterSize);
void filterFree();
void pathtrace(int frame, int iteration);
void denoiseImage(int num_iters, float c_phi, float n_phi, float p_phi);
std::vector<glm::vec3> getDenoisedImage();
void showGBuffer(uchar4 *pbo, int buffer_type);
void showImage(uchar4 *pbo, int iter, bool denoised);
