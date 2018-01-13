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

#include <host_defines.h>
#include <glm/vec3.hpp>
#include <device_launch_parameters.h>
#include "../../Particles.hpp"
#include "bodysystemcuda.h"
#include "../../util/helper_cuda.h"
#include <cooperative_groups.h>


__device__ glm::vec3
bodyBodyInteraction(glm::vec3 accel,
                    glm::vec4 pPos_and_radius1, glm::vec4 pVelo_and_mass1, Particles::TYPE type1,
                    glm::vec4 pPos_and_radius2, glm::vec4 pVelo_and_mass2, Particles::TYPE type2) {

    struct ParticleConst { //TODO as cuda symbol
        float elasticSpringConstant;
        float shellDepthFraction;
        float inelasticSpringForceReductionFactor;
    }
            ironConst = {.elasticSpringConstant = IRON_elasticSpringConstant,
            .shellDepthFraction = IRON_shellDepthFraction,
            .inelasticSpringForceReductionFactor = IRON_inelasticSpringForceReductionFactor},

            silicateConst = {.elasticSpringConstant = SILICATE_elasticSpringConstant,
            .shellDepthFraction = SILICATE_shellDepthFraction,
            .inelasticSpringForceReductionFactor = SILICATE_inelasticSpringForceReductionFactor};


    typename glm::vec3 r;
    glm::vec3 force(0, 0, 0);


    glm::vec3 difference = glm::vec3(pPos_and_radius1) - glm::vec3(pPos_and_radius2);

    float distance = glm::length(difference);
    glm::vec3 differenceNormal = (difference / distance);

    // Prevent numerical errors from divide by zero
    if (distance < distanceEpsilon) {
        //printf("The repulsive parameters were not set strong enough!\n");
        distance = distanceEpsilon;
    }

    // Newtonian gravity (F_g = -G m_1 m_2 r^(-2) hat{n})
    force -= differenceNormal * (float) ((double) G *
                                         (((double) pVelo_and_mass1.w *
                                           (double) pVelo_and_mass2.w) /
                                          ((double) (distance * distance))));

    // Separation "spring"
    if (distance < pPos_and_radius1.w + pPos_and_radius2.w) {
        ParticleConst pConst1 = (type1 == Particles::TYPE::IRON) ? ironConst : silicateConst;
        ParticleConst pConst2 = (type2 == Particles::TYPE::IRON) ? ironConst : silicateConst;

        float elasticConstantParticle1 = pConst1.elasticSpringConstant;
        float elasticConstantParticle2 = pConst2.elasticSpringConstant;

        // If the separation increases, i.e. the separation velocity is positive
        if (dot(differenceNormal, glm::vec3(pVelo_and_mass1)
                                  - glm::vec3(pVelo_and_mass2)) > 0) {
            // Check if the force shall be reduced due to plastic deformation
            if (distance < pPos_and_radius1.w * pConst1.shellDepthFraction +
                           pPos_and_radius2.w * pConst2.shellDepthFraction) {
                //case 1d
                elasticConstantParticle1 *= pConst1.inelasticSpringForceReductionFactor;
                elasticConstantParticle2 *= pConst2.inelasticSpringForceReductionFactor;
            } else if (distance <
                       pPos_and_radius1.w * pConst1.shellDepthFraction +
                       pPos_and_radius2.w) {
                // case 1c 1
                elasticConstantParticle1 *= pConst1.inelasticSpringForceReductionFactor;
            } else if (distance <
                       pPos_and_radius1.w +
                       pPos_and_radius2.w * pConst2.shellDepthFraction) {
                //case 1c 2
                elasticConstantParticle2 *= pConst2.inelasticSpringForceReductionFactor;
            }
        }

        float efficientSpringConstant = (elasticConstantParticle1 + elasticConstantParticle2);
        // Add compression force (F_s = k_eff ((r_1+r_2)^2 - r^2))
        force += differenceNormal * efficientSpringConstant *
                 ((pPos_and_radius1.w + pPos_and_radius2.w) *
                  (pPos_and_radius1.w + pPos_and_radius2.w) -
                  distance * distance);
    }


    // Leapfrog integration (better than Euler for gravity simulations)
    return force / pVelo_and_mass1.w; // a_i+1 = F_i+1 / m
}

struct SharedMemory_vec4 {
    __device__ inline operator glm::vec4 *() {
        extern __shared__ int __smem[];
        return (glm::vec4 *) __smem;
    }

    __device__ inline operator const glm::vec4 *() const {
        extern __shared__ int __smem[];
        return (glm::vec4 *) __smem;
    }
};

struct SharedMemory_type {
    __device__ inline operator Particles::TYPE *() {
        extern __shared__ int __smem[];
        return (Particles::TYPE *) __smem;
    }

    __device__ inline operator const Particles::TYPE *() const {
        extern __shared__ int __smem[];
        return (Particles::TYPE *) __smem;
    }
};


__device__ glm::vec3 computeBodyAccel(int pIdx,
                                      glm::vec4 *allParticlePos,
                                      Particles::Particles_cuda *particles,
                                      int numTiles,
                                      cooperative_groups::thread_block cta) {

    glm::vec4 *sharedPos_radius = SharedMemory_vec4();
    glm::vec4 *sharedVelo_mass = SharedMemory_vec4();
    Particles::TYPE *sharedType = SharedMemory_type();

    glm::vec4 pPos_radius = allParticlePos[pIdx];
    glm::vec4 pVelo_mass = particles->velo__mass[pIdx];
    Particles::TYPE pType = particles->type[pIdx];

    glm::vec3 accel = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < numTiles; tile++) {
        sharedPos_radius[threadIdx.x] = allParticlePos[tile * blockDim.x + threadIdx.x];
        sharedVelo_mass[threadIdx.x] = particles->velo__mass[tile * blockDim.x + threadIdx.x];
        sharedType[threadIdx.x] = particles->type[tile * blockDim.x + threadIdx.x];


        cooperative_groups::sync(cta);

        // This is the "tile_calculation" from the GPUG3 article.
#pragma unroll 128

        for (unsigned int counter = 0; counter < blockDim.x; counter++) {
            accel = bodyBodyInteraction(accel,
                                        pPos_radius, pVelo_mass, pType,
                                        sharedPos_radius[counter], sharedVelo_mass[counter], sharedType[counter]);
        }

        cooperative_groups::sync(cta);
    }

    return accel;
}


BodySystemCUDA::BodySystemCUDA(Particles *particles, cudaGraphicsResource_t particlePos) :
        cudaParticlePositionBuffer(particlePos) {
    numParticles = particles->numParticles;
    p_cuda = particles->to_cuda();
}

//__global__ void update_step(glm::vec4 *p_pos_radius_read,
//                            glm::vec4 *p_pos_radius_write,
//                            Particles::Particles_cuda *particles) {
//    int index = blockIdx.x * blockDim.x + threadIdx.x;
//
//    typename glm::vec3 accel = computeBodyAccel(index,
//                                                p_pos_radius_read, particles);
//
//}
//
//
//}

__global__ void
integrateBodies(glm::vec4 *allParticlePos_read,
                glm::vec4 *allParticlePos_write,
                Particles::Particles_cuda *particles,
                float deltaTime,
                int numTiles) {
    // Handle to thread block group
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
    int index = blockIdx.x * blockDim.x + threadIdx.x;

//    if (index >= deviceNumBodies) {
//        return;
//    }

    glm::vec3 accel = computeBodyAccel(index,
                                       allParticlePos_read,
                                       particles,
                                       numTiles,
                                       cta);

    allParticlePos_write[index] =
            allParticlePos_read[index] + glm::vec4(glm::vec3(particles->velo__mass[index]) * deltaTime +
                                                   accel * 0.5f * deltaTime *
                                                   deltaTime, 0); // x_i+1 = v_i*dt + a_i*dt^2/2
    particles->velo__mass[index] +=
            glm::vec4(accel * 0.5f * deltaTime, 0);
}

void BodySystemCUDA::updateStep(int numTimeSteps) {

    float deltaTime = numTimeSteps * 10.0f;
    size_t size = numParticles * sizeof(glm::vec4);

    checkCudaErrors(cudaGraphicsMapResources(1, &cudaParticlePositionBuffer));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **) &pPos[1 - currentRead],
                                                         &size, cudaParticlePositionBuffer));

    int blockSize = 256;

    int numBlocks = (numParticles + blockSize - 1) / blockSize;
    int numTiles = (numParticles + blockSize - 1) / blockSize;
    int sharedMemSize = blockSize * sizeof(glm::vec4) +
                        blockSize * sizeof(glm::vec4) +
                        blockSize * sizeof(Particles::TYPE); // 4 floats for pos

    // Update the position of the particles
    integrateBodies << < numBlocks, blockSize, sharedMemSize >> >
                                               (pPos[1 - currentRead],
                                                       pPos[1 - currentRead],
                                                       p_cuda,
                                                       deltaTime,
                                                       numTiles);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
//    cudaDeviceSynchronize();
    // Unmap the SSBO to be available to OpenGL
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaParticlePositionBuffer));
    currentRead = currentRead;

}