#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo, const int& show_idx);
void showImage(uchar4 *pbo, int iter, bool if_denoise);

void deNoise(
	const int& iteration,
	const float& ui_normalWeight,
	const float& ui_positionWeight,
	const float& ui_colorWeight);
void getVariance(const GBufferPixel* dev_gBuffer);
