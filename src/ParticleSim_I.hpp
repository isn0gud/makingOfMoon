#pragma once

#include "Particles.hpp"
#include "common.hpp"

class ParticleSim_I {

public:
    virtual void updateStep(int numTimeSteps)=0;
};
