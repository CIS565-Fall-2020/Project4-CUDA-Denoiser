#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo, const int& show_idx);
void showImage(uchar4 *pbo, int iter, bool if_denoise);

void deNoise(const int& iteration);
void getVariance(const GBufferPixel* dev_gBuffer);
