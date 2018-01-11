#include "PlanetBuilder.hpp"

#define PI 3.14159265359
#define estimatedPackingEfficiency 0.74 // packing efficency of a ffc-lattice

void PlanetBuilder::buildPlanet(Particles *particles, int startIdx, int numParticlesInPlanet,
                                Particles::TYPE coreType, float coreRadius,
                                Particles::TYPE outerLayerType, float radius,
                                glm::vec3 position, glm::vec3 velocity, glm::vec3 angularVelocity) {
    float planetVolume = 4 * PI * radius * radius * radius / 3;
    float particleVolume = estimatedPackingEfficiency * planetVolume / numParticlesInPlanet;
    float particleRadius = pow(3*particleVolume / (4*PI), 1.0f/3.0f);

    // Construct an fcc-lattice
    float latticeParameter = 2 * sqrt(2) * particleRadius;
    int girdSize = ceil(radius / latticeParameter)+1;

    glm::vec3 offsets[4] = {glm::vec3(0, 0, 0),
                            glm::vec3(latticeParameter/2, latticeParameter/2, 0),
                            glm::vec3(latticeParameter/2, 0, latticeParameter/2),
                            glm::vec3(0, latticeParameter/2, latticeParameter/2)};

    int i = startIdx;
    for(int x = -girdSize; x <= girdSize; x++)
    {
        for(int y = -girdSize; y <= girdSize; y++)
        {
            for(int z = -girdSize; z <= girdSize; z++)
            {
                glm::vec3 fccCellPos = glm::vec3(x, y, z) * latticeParameter;
                for(glm::vec3 offset : offsets)
                {
                    if(i - startIdx >= numParticlesInPlanet)
                        goto breakLoops;
                    glm::vec3 particlePositionInPlanet = fccCellPos + offset;
                    if(glm::dot(particlePositionInPlanet, particlePositionInPlanet) <= radius*radius) // is inside planet?
                    {
                        if(glm::dot(particlePositionInPlanet, particlePositionInPlanet) <= coreRadius*coreRadius) // is inside core?
                            particles->setParticleType(i, coreType, particleRadius, 1 / estimatedPackingEfficiency);
                        else
                            particles->setParticleType(i, outerLayerType, particleRadius, 1 / estimatedPackingEfficiency);

                        particles->pos[i] = glm::vec4(position + particlePositionInPlanet, 1);
                        particles->velo[i] = glm::vec4(velocity + glm::cross(angularVelocity, particlePositionInPlanet), 0); // v_rot = omega x r
                        i++;
                    }

                }
            }
        }
    }
    breakLoops:
    if(i < numParticlesInPlanet)
        buildPlanet(particles, startIdx, numParticlesInPlanet*1.05, coreType, coreRadius, outerLayerType, radius, position, velocity, angularVelocity);
}

glm::vec3 PlanetBuilder::sampleRandomPointInSphericalShell(float innerRadius, float outerRadius) {
    float innerRadiusCubed = innerRadius * innerRadius * innerRadius;
    float outerRadiusCubed = outerRadius * outerRadius * outerRadius;
    float radius = pow(innerRadiusCubed +
                       (outerRadiusCubed - innerRadiusCubed)
                       * randomFloatFromZeroToOne(),
                       1.f / 3.f);
    float mu = 1 - 2 * randomFloatFromZeroToOne();
    float planarLength = std::sqrt(1 - mu * mu);
    float azimuth = 2 * PI * randomFloatFromZeroToOne();
    return glm::vec3(radius * planarLength * (float) cos(azimuth), radius * planarLength * (float) sin(azimuth),
                     radius * mu);
}

float PlanetBuilder::randomFloatFromZeroToOne() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}


