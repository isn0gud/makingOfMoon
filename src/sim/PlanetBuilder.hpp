#ifndef AGP_PROJECT_PLANETBUILDER_HPP
#define AGP_PROJECT_PLANETBUILDER_HPP

#include "../common.hpp"
#include "Particle.hpp"

class PlanetBuilder
{
private:
    PlanetBuilder() {}
    ~PlanetBuilder() {}

    static float randomFloatFromZeroToOne();
    static glm::vec3 sampleRandomPointInSphericalShell(float innerRadius, float outerRadius);
public:
    static void buildPlanet(int numberOfParticles,
                            Particle::TYPE coreType, float coreRadius,
                            Particle::TYPE outerLayerType, float radius,
                            glm::vec3 position, glm::vec3 velocity, glm::vec3 angularVelocity,
                            std::vector<Particle*> &outParticles);
};

#endif //AGP_PROJECT_PLANETBUILDER_HPP
