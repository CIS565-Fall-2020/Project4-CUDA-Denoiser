#pragma once

extern GLuint pbo;

std::string currentTimeString();
bool init();
void mainLoop();

void save_img_from_frame(
    const char* show_item,
    const RenderState* renderState,
    const std::string& startTimeString,
    const int& samples
);