//
// Created by Pius Friesch on 19/11/2017.
//

#ifndef ASS_OPENGL_PARTICLECOMPUTATIONI_HPP
#define ASS_OPENGL_PARTICLECOMPUTATIONI_HPP

#include "Particle.hpp"
#include "../common.hpp"

class ParticleSimI {


public:


    virtual std::vector<Particle *> getParticles()= 0;

    void updateStep(int numTimeSteps) {
        updateStep(numTimeSteps, 1);
    }

    virtual void updateStep(int numTimeSteps, float stepSize)= 0;

};

#endif //ASS_OPENGL_PARTICLECOMPUTATIONI_HPP
