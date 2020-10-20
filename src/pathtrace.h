#pragma once

#include <vector>
#include "scene.h"
#include "utilities.h"

enum class BufferType {
	DirectSpecularIllumination,
	DirectSpecularIlluminationVariance,
	IndirectDiffuseIllumination,
	IndirectDiffuseIlluminationVariance,
	FullIllumination,
	Normal,
	Position,
	FilteredDirectSpecular,
	FilteredIndirectDiffuse,
	FilteredColor
};

void pathtraceInit(Scene *scene, int sqrtStratifiedSamples);
void pathtraceFree();
void updateStratifiedSamples(
	const std::vector<std::vector<IntersectionSample>> &samplers, const std::vector<CameraSample> &camSamples
);

void pathtrace(int frame, int iteration, int directLight, int numLights);
void aTrousPrepare(int iter);
void aTrous(
	BufferType buf, int levels, float radius, int iter, float colorWeight, float normalWeight, float positionWeight
);

void saveBufferState(BufferType buf, int iteration);
void sendBufferToPbo(uchar4 *pbo, BufferType buf, int iter);
