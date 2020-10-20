#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene, const std::vector<glm::vec3>& vertices, const std::vector<glm::vec3>& normals, int numVertices, const Geom& meshBB);
void pathtraceFree();
// void pathtrace(uchar4 *pbo, int frame, int iteration, int samplesPerPixel);
void pathtrace(int frame, int iteration, int samplesPerPixel);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);
void showDenoise(uchar4* pbo, int iter, float c_phi, float n_phi, float p_phi, int ui_filterSize);