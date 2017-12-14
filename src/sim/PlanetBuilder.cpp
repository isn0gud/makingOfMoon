#include "PlanetBuilder.hpp"

#define PI 3.14159265359
#define estimatedPackingEfficiency 0.5

void PlanetBuilder::buildPlanet(int numberOfParticles,
                                Particle::TYPE coreType, float coreRadius,
                                Particle::TYPE outerLayerType, float radius,
                                glm::vec3 position, glm::vec3 velocity, glm::vec3 angularVelocity,
                                std::vector<Particle*> &outParticles)
{
    int oldPartilcesSize = outParticles.size();
    outParticles.resize(oldPartilcesSize+ numberOfParticles);

    float volumePerParticleInCore = estimatedPackingEfficiency * 4*PI*coreRadius*coreRadius*coreRadius/3 / numberOfParticles;

    float coreParticleRadius = pow(3*volumePerParticleInCore/(4*PI), 1.0f/3.0f);

    float volumePerParticleInOuterLayer = estimatedPackingEfficiency * (4*PI*radius*radius*radius/3) / (numberOfParticles);
    float outerLayerParticleRadius = pow(3*volumePerParticleInOuterLayer/(4*PI), 1.0f/3.0f);

    std::cout << volumePerParticleInOuterLayer << std::endl;
    std::cout << outerLayerParticleRadius << std::endl;

    float particleRadius = 188.7;

    for(int i = oldPartilcesSize; i < oldPartilcesSize + numberOfParticles; i++)
    {
        glm::vec3 offsetFromCenterOfPlanet = sampleRandomPointInSphericalShell(0, radius);

        if(glm::length(offsetFromCenterOfPlanet) < coreRadius)
            outParticles[i] = Particle::particleFromType(coreType, particleRadius);
        else
            outParticles[i] = Particle::particleFromType(outerLayerType, particleRadius);
        outParticles[i]->pos = position + offsetFromCenterOfPlanet;
        outParticles[i]->velo = velocity + glm::cross(angularVelocity, offsetFromCenterOfPlanet);
    }
}

glm::vec3 PlanetBuilder::sampleRandomPointInSphericalShell(float innerRadius, float outerRadius)
{
    float innerRadiusCubed = innerRadius*innerRadius*innerRadius;
    float outerRadiusCubed = outerRadius*outerRadius*outerRadius;
    float radius = pow(innerRadiusCubed + (outerRadiusCubed - innerRadiusCubed)*randomFloatFromZeroToOne(), 1.f/3.f);
    float mu = 1 - 2*randomFloatFromZeroToOne();
    float planarLength = sqrt(1- mu*mu);
    float azimuth = 2*PI*randomFloatFromZeroToOne();
    return glm::vec3(radius*planarLength*(float)cos(azimuth), radius*planarLength*(float)sin(azimuth), radius*mu);
}

float PlanetBuilder::randomFloatFromZeroToOne()
{
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}
