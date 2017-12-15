//
// Created by Karl Kvarnfors on 26/11/2017.
//

#ifndef AGP_PROJECT_GRAVITYSIM_HPP
#define AGP_PROJECT_GRAVITYSIM_HPP

#include "ParticleSimI.hpp"

class GravitySim : public ParticleSimI {
private:
    std::vector<Particle *> particles;
    std::vector<glm::vec3> forces;

    glm::vec3 getVelocityFromRotation(glm::vec3 angularVelocity, glm::vec3 position, glm::vec3 centerOfRotation);

public:
    GravitySim();

    const std::vector<Particle *>& getParticles() override;

    void updateStep(int numTimeSteps) override;

    virtual ~GravitySim();
};


#endif //AGP_PROJECT_GRAVITYSIM_HPP
