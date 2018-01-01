#pragma once

#include "Particles.hpp"
#include "common.hpp"

class ParticleSimI {

public:

    virtual void initParticles(Particles *particles)=0;

    virtual void updateStep(int numTimeSteps)=0;
};
