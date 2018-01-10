#include "GravitySimGPU.cuh"
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
//#include "../../cudaUtil.cuh"
#include "vector_types.h"
#include "../../util/helper_cuda.h"

#define timeStep 10.0f
#define COLL_SPEED 1.5
#define CUBE_SIDE 5

#define BLOCK_SIZE 256

//  units: SI, but km instead of m
//  6.674×10−20 (km)^3⋅kg^(−1)⋅s^(−2)
#define G 6.674E-20
#define distanceEpsilon 47.0975

__global__ void update_step(glm::vec4 *pPos, Particles::Particles_cuda *particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    particles->numParticles = 5000; //TODO add numParticles as parameter
    if (i < particles->numParticles) {
        //TODO add update steps
        for (int j = i + 1; j < particles->numParticles; j++) {
            glm::vec3 force(0, 0, 0);
            glm::vec3 difference = pPos[i] - pPos[j];

            float distance = glm::length(difference);
            glm::vec3 differenceNormal = (difference / distance);

            // Prevent numerical errors from divide by zero
            if (distance < distanceEpsilon) {
                //printf("The repulsive parameters were not set strong enough!\n");
                distance = distanceEpsilon;
            }

            // Newtonian gravity (F_g = -G m_1 m_2 r^(-2) hat{n}), doubles are needed to prevent overflow, needs to be fixed in GPU implementation
            force -= differenceNormal * (float) ((double) G *
                                                 (((double) particles->mass[i] * (double) particles->mass[j]) /
                                                  ((double) (distance * distance))));

            // Separation "spring"
            if (distance < particles->radius[i] + particles->radius[j]) {
                float elasticConstantParticle1 = particles->elasticSpringConstant[i];
                float elasticConstantParticle2 = particles->elasticSpringConstant[j];

                // If the separation increases, i.e. the separation velocity is positive
                if (dot(differenceNormal, glm::vec3(particles->velo[i] - particles->velo[j])) > 0) {
                    // Check if the force shall be reduced due to plastic deformation
                    if (distance < particles->radius[i] * particles->shellDepthFraction[i] +
                                   particles->radius[j] * particles->shellDepthFraction[j]) {
                        //case 1d
                        elasticConstantParticle1 *= particles->inelasticSpringForceReductionFactor[i];
                        elasticConstantParticle2 *= particles->inelasticSpringForceReductionFactor[j];
                    } else if (distance <
                               particles->radius[i] * particles->shellDepthFraction[i] + particles->radius[j]) {
                        // case 1c 1
                        elasticConstantParticle1 *= particles->inelasticSpringForceReductionFactor[i];
                    } else if (distance <
                               particles->radius[i] + particles->radius[j] * particles->shellDepthFraction[j]) {
                        //case 1c 2
                        elasticConstantParticle2 *= particles->inelasticSpringForceReductionFactor[j];
                    }
                }

                float efficientSpringConstant = (elasticConstantParticle1 + elasticConstantParticle2);
                // Add compression force (F_s = k_eff ((r_1+r_2)^2 - r^2))
                force += differenceNormal * efficientSpringConstant *
                         ((particles->radius[i] + particles->radius[j]) *
                          (particles->radius[i] + particles->radius[j]) -
                          distance * distance);
            }

            // Leapfrog integration (better than Euler for gravity simulations)
            glm::vec4 newAcceleration = glm::vec4(force / particles->mass[i], 1); // a_i+1 = F_i+1 / m
            pPos[i] += particles->velo[i] * timeStep +
                       particles->accel[i] * 0.5f * timeStep *
                       timeStep; // x_i+1 = v_i*dt + a_i*dt^2/2
            particles->velo[i] +=
                    (particles->accel[i] + newAcceleration) * 0.5f * timeStep; // v_i+1 = v_i + (a_i + a_i+1)dt/2
            particles->accel[i] = newAcceleration;
        }
    }
}


void GravitySimGPU::updateStep(int numTimeSteps) {

    size_t size = numParticles * sizeof(glm::vec4);
    glm::vec4 *d_particles;

    checkCudaErrors(cudaGraphicsMapResources(1, &vboParticlesPos_cuda));

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **) &d_particles,
                                                         &size, vboParticlesPos_cuda));

    int numberOfBlocks = (numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Update the position of the particles
    update_step<<<numberOfBlocks, BLOCK_SIZE>>>(d_particles, p_cuda);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    // Unmap the SSBO to be available to OpenGL
    checkCudaErrors(cudaGraphicsUnmapResources(1, &vboParticlesPos_cuda));
}

GravitySimGPU::GravitySimGPU(Particles *particles, cudaGraphicsResource_t particlePos) :
        vboParticlesPos_cuda(particlePos) {
    numParticles = particles->numParticles;
    p_cuda = particles->to_cuda();
}
