#include "PlanetBuilder.hpp"

#define PI 3.14159265359f
#define estimatedPackingEfficiency 0.74f // packing efficency of a ffc-lattice

Particles::ParticlesInit PlanetBuilder::buildPlanet(int numParticlesInPlanet,
                                                    Particles::TYPE coreType, float coreRadius,
                                                    Particles::TYPE outerLayerType, float planetRadius,
                                                    glm::vec3 position, glm::vec3 velocity, glm::vec3 angularVelocity) {

    Particles::ParticlesInit particlesInit;

    int i;
    bool nicePlanetGenerated = false;
    while (!nicePlanetGenerated) {
        i = 0;
        particlesInit.clear();

        float planetVolume = 4.0f * PI * planetRadius * planetRadius * planetRadius / 3.0f;
        float particleVolume = estimatedPackingEfficiency * planetVolume / numParticlesInPlanet;
        float particleRadius = pow(3.0f * particleVolume / (4.0f * PI), 1.0f / 3.0f);
        // Construct an fcc-lattice
        float latticeParameter = 2.0f * (float) sqrt(2.0f) * particleRadius;
        int girdSize = ceil(planetRadius / latticeParameter) + 1;

        glm::vec3 offsets[4] = {glm::vec3(0, 0, 0),
                                glm::vec3(latticeParameter / 2, latticeParameter / 2, 0),
                                glm::vec3(latticeParameter / 2, 0, latticeParameter / 2),
                                glm::vec3(0, latticeParameter / 2, latticeParameter / 2)};


        for (int x = -girdSize; x <= girdSize; x++) {
            for (int y = -girdSize; y <= girdSize; y++) {
                for (int z = -girdSize; z <= girdSize; z++) {
                    glm::vec3 fccCellPos = glm::vec3(x, y, z) * latticeParameter;
                    for (glm::vec3 offset : offsets) {
                        if (i >= numParticlesInPlanet) {
                            // try again with more particles since not the number of particles couldn't fill the lattice
                            goto breakLoops; // more clear than 4 breaks
                        }
                        glm::vec3 particlePositionInPlanet = fccCellPos + offset;
                        if (glm::dot(particlePositionInPlanet, particlePositionInPlanet) <=
                            planetRadius * planetRadius) // is inside planet?
                        {
                            particlesInit.pos__radius.push_back(glm::vec4(position + particlePositionInPlanet, -1));
                            particlesInit.velo__mass.push_back(glm::vec4(
                                    velocity + glm::cross(angularVelocity, particlePositionInPlanet),
                                    -1)); // v_rot = omega x r
                            //set type, radius and mass
                            if (glm::dot(particlePositionInPlanet, particlePositionInPlanet) <=
                                coreRadius * coreRadius) // is inside core?
                                particlesInit.setParticleType(i, coreType, particleRadius,
                                                              1.0f / estimatedPackingEfficiency);
                            else
                                particlesInit.setParticleType(i, outerLayerType, particleRadius,
                                                              1.0f / estimatedPackingEfficiency);
                            i++;
                        }

                    }
                }
            }
        }
        breakLoops:
        particlesInit.numParticles = i;
        if (i < numParticlesInPlanet) {
            //try again with more particles
            numParticlesInPlanet *= 1.05f;
            nicePlanetGenerated = false;
        } else {
            nicePlanetGenerated = true;
        }
    };
    return particlesInit;
}

glm::vec3 PlanetBuilder::sampleRandomPointInSphericalShell(float innerRadius, float outerRadius) {
    float innerRadiusCubed = innerRadius * innerRadius * innerRadius;
    float outerRadiusCubed = outerRadius * outerRadius * outerRadius;
    float planetRadius = pow(innerRadiusCubed +
                             (outerRadiusCubed - innerRadiusCubed)
                             * randomFloatFromZeroToOne(),
                             1.f / 3.f);
    float mu = 1 - 2 * randomFloatFromZeroToOne();
    float planarLength = std::sqrt(1 - mu * mu);
    float azimuth = 2 * PI * randomFloatFromZeroToOne();
    return glm::vec3(planetRadius * planarLength * (float) cos(azimuth),
                     planetRadius * planarLength * (float) sin(azimuth),
                     planetRadius * mu);
}

float PlanetBuilder::randomFloatFromZeroToOne() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}


