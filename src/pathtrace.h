#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

// Denoiser additions
void setDenoise(bool val);
void setFilterSize(int val);
void setColorWeight(float val);
void setNormalWeight(float val);
void setPositionWeight(float val);

void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);
void showDenoisedImage(uchar4* pbo, int iter);