#pragma once

#include "../../ParticleSim_I.hpp"
#include <driver_types.h>

class GravitySimGPU : ParticleSim_I {

private:
    int numParticles;
    Particles::Particles_cuda *p_cuda;
    cudaGraphicsResource_t cudaParticlePositionBuffer;

public:
    GravitySimGPU(Particles *particles, cudaGraphicsResource_t particlePos);

    void updateStep(int numTimeSteps) override;
};




