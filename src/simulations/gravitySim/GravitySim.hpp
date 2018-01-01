#pragma once

#include "../../ParticleSimI.hpp"

class GravitySim : public ParticleSimI {
private:
    Particles *particles = nullptr;
    std::vector<glm::vec3> forces;

//    glm::vec3 getVelocityFromRotation(glm::vec3 angularVelocity, glm::vec3 position, glm::vec3 centerOfRotation);

public:
    void initParticles(Particles *particles) override;


//    const std::vector<Particle *>& getParticles() override;

    void updateStep(int numTimeSteps) override;

    virtual ~GravitySim();
};


