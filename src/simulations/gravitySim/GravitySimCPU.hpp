#pragma once

#include "../../ParticleSim_I.hpp"
#include "../../Particles.hpp"

class GravitySimCPU : public ParticleSim_I {
private:
    Particles *particles = nullptr;

public:
    GravitySimCPU(Particles *particles);

    void updateStep(int numTimeSteps) override;

    virtual ~GravitySimCPU();
};


