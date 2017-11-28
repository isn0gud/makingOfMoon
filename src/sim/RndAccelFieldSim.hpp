//
// Created by Pius Friesch on 20/11/2017.
//

#ifndef ASS_OPENGL_RNDACCELFIELDSIM_HPP
#define ASS_OPENGL_RNDACCELFIELDSIM_HPP

#define NUM_PARTICLES 100
#define SIM_SPEED 100.0f

#include "ParticleSimI.hpp"

class RndAccelFieldSim : public ParticleSimI {
private:


    std::vector<Particle *> particles;


public:
    RndAccelFieldSim();

    std::vector<Particle *> getParticles() override;

    void updateStep(int numTimeSteps, float stepSize) override;

private:
    glm::vec3 gravityAccelToCenter(glm::vec3 pos) const;

};


#endif //ASS_OPENGL_RNDACCELFIELDSIM_HPP
