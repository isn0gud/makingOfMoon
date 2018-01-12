#pragma once

#include "common.hpp"

class ParticleSim_I {

public:
    virtual void updateStep(int numTimeSteps)=0;
};
