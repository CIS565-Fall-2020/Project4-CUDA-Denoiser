#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration, bool denoise, float c_phi, float n_phi, float p_phi);
void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter);
