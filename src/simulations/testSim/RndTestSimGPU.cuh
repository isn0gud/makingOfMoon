#pragma once

#include <driver_types.h>
#include "../../ParticleSim_I.hpp"
#include "../../Particles.hpp"

class RndTestSimGPU : ParticleSim_I {

private:
    Particles *particles;
    cudaGraphicsResource_t vboParticlesPos_cuda;

public:
    RndTestSimGPU(Particles *particles, cudaGraphicsResource_t particlePos);

    void updateStep(int numTimeSteps) override;

};



