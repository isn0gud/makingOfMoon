#include "RndTestSimGPU.cuh"
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "../../util/helper_cuda.h"

__global__ void update(glm::vec4 *pPos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NUM_PARTICLES) {
        pPos[i] = pPos[i] + glm::vec4(0.01 * pPos[i].x, 0.01 * pPos[i].y,
                                      0.01 * pPos[i].z, 1);
//        printf("test\n");
    }
}


void RndTestSimGPU::updateStep(int numTimeSteps) {

    size_t size = particles->numParticles * sizeof(glm::vec4);
    glm::vec4 *d_particles;

    checkCudaErrors(cudaGraphicsMapResources(1, &vboParticlesPos_cuda));

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **) &d_particles,
                                                   &size, vboParticlesPos_cuda));


    // Update the position of the particles
    update <<< 256, 256 >>> (d_particles);

    checkCudaErrors(cudaDeviceSynchronize());

    // Unmap the SSBO to be available to OpenGL
    checkCudaErrors(cudaGraphicsUnmapResources(1, &vboParticlesPos_cuda));
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

RndTestSimGPU::RndTestSimGPU(Particles *particles, cudaGraphicsResource_t particlePos)
        : particles(particles),
          vboParticlesPos_cuda(particlePos) {}

