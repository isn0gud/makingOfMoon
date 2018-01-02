#pragma once

#include "../../ParticleSim_I.hpp"

class GravitySimCPU : public ParticleSim_I {
private:
    Particles *particles = nullptr;
    std::vector<glm::vec3> forces;

//    glm::vec3 getVelocityFromRotation(glm::vec3 angularVelocity, glm::vec3 position, glm::vec3 centerOfRotation);

public:
    GravitySimCPU(Particles *particles);

    void updateStep(int numTimeSteps) override;

    virtual ~GravitySimCPU();
};


