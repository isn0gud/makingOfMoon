#pragma once

#include "../common.hpp"
#include "../Particles.hpp"

class PlanetBuilder {
private:
    PlanetBuilder() {}

    ~PlanetBuilder() {}

    static float randomFloatFromZeroToOne();
    static glm::vec3 sampleRandomPointInSphericalShell(float innerRadius, float outerRadius);

public:
    static void buildPlanet(Particles *particles, int startIdx, int numParticlesInPlanet, Particles::TYPE coreType,
                            float coreRadius,
                            Particles::TYPE outerLayerType, float radius, glm::vec3 position, glm::vec3 velocity,
                            glm::vec3 angularVelocity);
};

