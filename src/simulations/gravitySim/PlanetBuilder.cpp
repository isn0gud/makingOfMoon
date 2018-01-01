#include "PlanetBuilder.hpp"

#define PI 3.14159265359
#define estimatedPackingEfficiency 0.5


void PlanetBuilder::buildPlanet(Particles *particles, Particles::TYPE coreType, float coreRadius,
                                Particles::TYPE outerLayerType, float radius, glm::vec3 position, glm::vec3 velocity,
                                glm::vec3 angularVelocity) {

    float volumePerParticleInCore =
            estimatedPackingEfficiency * 4 * PI * coreRadius *
            coreRadius * coreRadius / 3 / particles->numParticles;

    float coreParticleRadius = pow(3 * volumePerParticleInCore /
                                   (4 * PI), 1.0f / 3.0f);

    float volumePerParticleInOuterLayer =
            estimatedPackingEfficiency *
            (4 * PI * radius * radius * radius / 3) /
            (particles->numParticles);

    float outerLayerParticleRadius = pow(3 * volumePerParticleInOuterLayer /
                                         (4 * PI), 1.0f / 3.0f);

    std::cout << volumePerParticleInOuterLayer << std::endl;
    std::cout << outerLayerParticleRadius << std::endl;

    float particleRadius = 188.7;

    for (int i = 0; i < particles->numParticles; i++) {
        glm::vec3 offsetFromCenterOfPlanet = sampleRandomPointInSphericalShell(0, radius);

        if (glm::length(offsetFromCenterOfPlanet) < coreRadius) {
            particles->setParticleType(i, coreType, particleRadius);
        } else {
            particles->setParticleType(i, outerLayerType, particleRadius);
        }

        particles->particlePos[i] = glm::vec4(position + offsetFromCenterOfPlanet, 1);
        particles->particleVelo[i] = glm::vec4(velocity + glm::cross(angularVelocity, offsetFromCenterOfPlanet), 1);
    }
}

glm::vec3 PlanetBuilder::sampleRandomPointInSphericalShell(float innerRadius, float outerRadius) {
    float innerRadiusCubed = innerRadius * innerRadius * innerRadius;
    float outerRadiusCubed = outerRadius * outerRadius * outerRadius;
    float radius = pow(innerRadiusCubed +
                       (outerRadiusCubed - innerRadiusCubed)
                       * randomFloatFromZeroToOne(),
                       1.f / 3.f);
    float mu = 1 - 2 * randomFloatFromZeroToOne();
    float planarLength = sqrt(1 - mu * mu);
    float azimuth = 2 * PI * randomFloatFromZeroToOne();
    return glm::vec3(radius * planarLength * (float) cos(azimuth), radius * planarLength * (float) sin(azimuth),
                     radius * mu);
}

float PlanetBuilder::randomFloatFromZeroToOne() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

