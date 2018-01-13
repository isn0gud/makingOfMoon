/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#pragma once

#include "../../ParticleSim_I.hpp"
#include "../../Particles.hpp"

class BodySystemCUDA : public ParticleSim_I {

private:
    int numParticles;

    glm::vec4 *pPos[2];
    int currentRead = 0;

    Particles::Particles_cuda *p_cuda;
    cudaGraphicsResource_t cudaParticlePositionBuffer;

public:
    BodySystemCUDA(Particles *particles, cudaGraphicsResource_t particlePos);

    void updateStep(int numTimeSteps) override;
};
