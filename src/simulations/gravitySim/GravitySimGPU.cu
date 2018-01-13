#include "GravitySimGPU.hpp"
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "vector_types.h"
#include "../../util/helper_cuda.h"

#define timeStep 2.f
#define COLL_SPEED 1.5
#define CUBE_SIDE 5

#define BLOCK_SIZE 256


__global__ void update_step(glm::vec4 *p_pos_radius, Particles::Particles_cuda *particles) {

    struct ParticleConst {
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


    int i = blockIdx.x * blockDim.x + threadIdx.x;

//    particles->numParticles = NUM_PARTICLES;
    if (i < *particles->numParticles) {
        glm::vec3 force(0, 0, 0);
        for (int j = 0; j < *particles->numParticles; j++) {
            if (i == j)
                continue;

            glm::vec3 difference = glm::vec3(p_pos_radius[i]) - glm::vec3(p_pos_radius[j]);

            float distance = glm::length(difference);
            glm::vec3 differenceNormal = (difference / distance);

            // Prevent numerical errors from divide by zero
            if (distance < distanceEpsilon) {
                //printf("The repulsive parameters were not set strong enough!\n");
                distance = distanceEpsilon;
            }

            // Newtonian gravity (F_g = -G m_1 m_2 r^(-2) hat{n})
            force -= differenceNormal * (float) ((double) G *
                                                 (((double) particles->velo__mass[i].w *
                                                   (double) particles->velo__mass[j].w) /
                                                  ((double) (distance * distance))));

            // Separation "spring"
            if (distance < p_pos_radius[i].w + p_pos_radius[j].w) {
                ParticleConst pConst1 = (particles->type[i] == Particles::TYPE::IRON) ? ironConst : silicateConst;
                ParticleConst pConst2 = (particles->type[j] == Particles::TYPE::IRON) ? ironConst : silicateConst;

                float elasticConstantParticle1 = pConst1.elasticSpringConstant;
                float elasticConstantParticle2 = pConst2.elasticSpringConstant;

                // If the separation increases, i.e. the separation velocity is positive
                if (dot(differenceNormal, glm::vec3(particles->velo__mass[i])
                                          - glm::vec3(particles->velo__mass[j])) > 0) {
                    // Check if the force shall be reduced due to plastic deformation
                    if (distance < p_pos_radius[i].w * pConst1.shellDepthFraction +
                                   p_pos_radius[j].w * pConst2.shellDepthFraction) {
                        //case 1d
                        elasticConstantParticle1 *= pConst1.inelasticSpringForceReductionFactor;
                        elasticConstantParticle2 *= pConst2.inelasticSpringForceReductionFactor;
                    } else if (distance <
                               p_pos_radius[i].w * pConst1.shellDepthFraction +
                               p_pos_radius[j].w) {
                        // case 1c 1
                        elasticConstantParticle1 *= pConst1.inelasticSpringForceReductionFactor;
                    } else if (distance <
                               p_pos_radius[i].w +
                               p_pos_radius[j].w * pConst2.shellDepthFraction) {
                        //case 1c 2
                        elasticConstantParticle2 *= pConst2.inelasticSpringForceReductionFactor;
                    }
                }

                float efficientSpringConstant = (elasticConstantParticle1 + elasticConstantParticle2);
                // Add compression force (F_s = k_eff ((r_1+r_2)^2 - r^2))
                force += differenceNormal * efficientSpringConstant *
                         ((p_pos_radius[i].w + p_pos_radius[j].w) *
                          (p_pos_radius[i].w + p_pos_radius[j].w) -
                          distance * distance);
            }
        }

        // Leapfrog integration (better than Euler for gravity simulations)
        glm::vec3 newAcceleration = force / particles->velo__mass[i].w; // a_i+1 = F_i+1 / m

        p_pos_radius[i] += glm::vec4(glm::vec3(particles->velo__mass[i]) * timeStep +
                                     newAcceleration * 0.5f * timeStep *
                                     timeStep, 0); // x_i+1 = v_i*dt + a_i*dt^2/2
        particles->velo__mass[i] +=
                glm::vec4(newAcceleration * 0.5f * timeStep, 0);
    }
}

void GravitySimGPU::updateStep(int numTimeSteps) {

    size_t size = numParticles * sizeof(glm::vec4);
    glm::vec4 *d_particles;

    checkCudaErrors(cudaGraphicsMapResources(1, &cudaParticlePositionBuffer));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **) &d_particles,
                                                         &size, cudaParticlePositionBuffer));

    int numberOfBlocks = (numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Update the position of the particles
    update_step << < numberOfBlocks, BLOCK_SIZE >> > (d_particles, p_cuda);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    // Unmap the SSBO to be available to OpenGL
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaParticlePositionBuffer));
}

GravitySimGPU::GravitySimGPU(Particles *particles, cudaGraphicsResource_t particlePos) :
        cudaParticlePositionBuffer(particlePos) {
    numParticles = particles->numParticles;
    p_cuda = particles->to_cuda();
}
