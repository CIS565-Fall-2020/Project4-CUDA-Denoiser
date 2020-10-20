#pragma once

#include "intersections.h"

/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

/*
* Computes sampled glossy reflection direction.
*/
__host__ __device__
glm::vec3 calculateImperfectSpecularDirection(
    glm::vec3 normal, float spec_exp, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    float theta = glm::acos(glm::pow(u01(rng), 1.f / (spec_exp + 1.f)));
    float phi = TWO_PI * u01(rng);
    glm::vec3 sample_dir(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));

    // Compute tangent space
    glm::vec3 tangent;
    glm::vec3 bitangent;

    if (glm::abs(normal.x) > glm::abs(normal.y)) {
        tangent = glm::vec3(-normal.z, 0.f, normal.x) / glm::sqrt(normal.x * normal.x + normal.z * normal.z);
    }
    else {
        tangent = glm::vec3(0, normal.z, -normal.y) / glm::vec3(normal.y * normal.y + normal.z * normal.z);
    }
    bitangent = glm::cross(normal, tangent);

    // Transform sample from specular space to tangent space
    glm::mat3 transform(tangent, bitangent, normal);
    return glm::normalize(transform * sample_dir);
}

/**
 * Simple ray scattering with diffuse, perfect specular, glossy and fresnel dielectric support.
 */
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    glm::vec3 newDirection;
    if (m.hasReflective) {
        if (m.specular.exponent > 0.f) {
            // Glossy
            newDirection = calculateImperfectSpecularDirection(normal, m.specular.exponent, rng);
        }
        else {
            // Mirror
            newDirection = glm::reflect(pathSegment.ray.direction, normal);
        }
    }
    else if (m.hasRefractive) {
        // Fresnel
        float cosine;
        float reflect_prob;
        bool entering = glm::dot(pathSegment.ray.direction, normal) < 0;
        float etaI = entering ? 1.f : m.indexOfRefraction;
        float etaT = entering ? m.indexOfRefraction : 1.f;
        float eta = etaI / etaT;
        cosine = entering ? -glm::dot(pathSegment.ray.direction, normal) / glm::length(pathSegment.ray.direction) :
            m.indexOfRefraction * glm::dot(pathSegment.ray.direction, normal) / glm::length(pathSegment.ray.direction);
        newDirection = glm::refract(pathSegment.ray.direction, normal, eta);
        if (glm::length(newDirection) == 0.f) {
            reflect_prob = 1.f;
        }
        else {
            float R0 = (etaI - etaT) / (etaI + etaT);
            R0 *= R0;
            reflect_prob = R0 + (1.f - R0) * glm::pow(1.f - cosine, 5.f);
        }

        thrust::uniform_real_distribution<float> u01(0, 1);
        float prob = u01(rng);
        if (prob < reflect_prob) {
            newDirection = glm::reflect(pathSegment.ray.direction, normal);
        }
    } 
    else {
        // Diffuse
        newDirection = calculateRandomDirectionInHemisphere(normal, rng);
    }

    pathSegment.ray.direction = newDirection;
    pathSegment.ray.origin = intersect + (newDirection * 0.0001f);
}
