#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
//void pathtrace(uchar4 *pbo, int frame, int iteration);
void pathtrace(int frame, int iter, float cPhi, float nPhi, float pPhi, int filterIter);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);
void showDenoisedImage(uchar4* pbo, int iter);
