//
// Created by Karl Kvarnfors on 26/11/2017.
//

#ifndef ASS_OPENGL_GRAVITYSIM_HPP
#define ASS_OPENGL_GRAVITYSIM_HPP

#include "ParticleSimI.hpp"

class GravitySim : public ParticleSimI {
private:
    std::vector<Particle *> particles;
    std::vector<glm::vec3> forces;

    glm::vec3 getVelocityFromRotation(glm::vec3 angularVelocity, glm::vec3 position, glm::vec3 centerOfRotation);

public:
    GravitySim();

    std::vector<Particle *> getParticles() override;

    void updateStep(int numTimeSteps, float dt) override;

    virtual ~GravitySim();
};


#endif //ASS_OPENGL_GRAVITYSIM_HPP
