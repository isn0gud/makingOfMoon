#pragma once

#include "../common.hpp"
#include "../Particles.hpp"
#include <vector>

class PlanetBuilder {
private:
    PlanetBuilder() {}

    ~PlanetBuilder() {}

    static float randomFloatFromZeroToOne();

    static glm::vec3 sampleRandomPointInSphericalShell(float innerRadius, float outerRadius);

public:
    static Particles::ParticlesInit buildPlanet( int numParticlesInPlanet, TYPE coreType,
                           float coreRadius,
                           TYPE outerLayerType, float radius, glm::vec3 position, glm::vec3 velocity,
                           glm::vec3 angularVelocity);


};

