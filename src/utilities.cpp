//  UTILITYCORE- A Utility Library by Yining Karl Li
//  This file is part of UTILITYCORE, Copyright (c) 2012 Yining Karl Li
//
//  File: utilities.cpp
//  A collection/kitchen sink of generally useful functions

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <iostream>
#include <cstdio>

#include "utilities.h"

float utilityCore::clamp(float f, float min, float max)
{
    if (f < min)
    {
        return min;
    } 
    else if (f > max)
    {
        return max;
    } 
    else {
        return f;
    }
}

bool utilityCore::replaceString(std::string& str, const std::string& from, const std::string& to)
{
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

std::string utilityCore::convertIntToString(int number) 
{
    std::stringstream ss;
    ss << number;
    return ss.str();
}

glm::vec3 utilityCore::clampRGB(glm::vec3 color)
{
    if (color[0] < 0) 
    {
        color[0] = 0;
    } else if (color[0] > 255) 
    {
        color[0] = 255;
    }
    if (color[1] < 0) 
    {
        color[1] = 0;
    } else if (color[1] > 255)
    {
        color[1] = 255;
    }
    if (color[2] < 0)
    {
        color[2] = 0;
    } else if (color[2] > 255)
    {
        color[2] = 255;
    }
    return color;
}

bool utilityCore::epsilonCheck(float a, float b) 
{
    if (fabs(fabs(a) - fabs(b)) < EPSILON)
    {
        return true;
    } else {
        return false;
    }
}

glm::mat4 utilityCore::buildTransformationMatrix(const glm::vec3& translation, const glm::vec3& rotation, const glm::vec3& scale)
{
    glm::mat4 translationMat  = glm::translate(glm::mat4(), translation);
    glm::mat4 rotationMat     = glm::rotate(glm::mat4(), rotation.x * (float) PI / 180, glm::vec3(1, 0, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * (float) PI / 180, glm::vec3(0, 1, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * (float) PI / 180, glm::vec3(0, 0, 1));
    glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
    return translationMat * rotationMat * scaleMat;
}

std::vector<std::string> utilityCore::tokenizeString(std::string str)
{
    std::stringstream strstr(str);
    std::istream_iterator<std::string> it(strstr);
    std::istream_iterator<std::string> end;
    std::vector<std::string> results(it, end);
    return results;
}

std::istream& utilityCore::safeGetline(std::istream& is, std::string& t) {
    t.clear();

    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks,
    // such as thread synchronization and updating the stream state.

    std::istream::sentry se(is, true);
    std::streambuf* sb = is.rdbuf();

    for (;;) 
    {
        int c = sb->sbumpc();
        switch (c) 
        {
        case '\n':
            return is;
        case '\r':
            if (sb->sgetc() == '\n')
                sb->sbumpc();
            return is;
        case EOF:
            // Also handle the case when the last line has no line ending
            if (t.empty())
                is.setstate(std::ios::eofbit);
            return is;
        default:
            t += (char)c;
        }
    }
}

void utilityCore::computeGaussianKernel(std::vector<float>& kernel, const int kernel_size, const float sigma)
{
    int half_size = kernel_size / 2;
    float sum = 0.f;
    float double_sigma_square = 2 * sigma * sigma;
    float alpha = 1.f / (PI * double_sigma_square);
    for (int i = -half_size; i <= half_size; i++)
    {
        for (int j = -half_size; j <= half_size; j++)
        {
            float temp = alpha * exp(-((i * i + j * j) / double_sigma_square));
            sum += temp;
            kernel.push_back(temp);
        }
    }

    for (int i = 0; i < kernel.size(); i++)
    {
        kernel[i] /= sum;
    }
}