#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo);
void showDenoise(uchar4* pbo, int iter, int filterIter, float cPhi, float nPhi, float pPhi);
void showImage(uchar4* pbo, int iter);
