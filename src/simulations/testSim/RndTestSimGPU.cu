#include "RndTestSimGPU.cuh"
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

__global__ void update(glm::vec4 *pPos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 3000) {
        pPos[i] = pPos[i] + glm::vec4(0.01 * pPos[i].x, 0.01 * pPos[i].y,
                                      0.01 * pPos[i].z, 1);
//        printf("test\n");
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
///src: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


void RndTestSimGPU::updateStep(int numTimeSteps) {

    size_t size = particles->numParticles * sizeof(glm::vec4);
    glm::vec4 *d_particles;

    gpuErrchk(cudaGraphicsMapResources(1, &vboParticlesPos_cuda));

    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void **) &d_particles,
                                                   &size, vboParticlesPos_cuda));


    // Update the position of the particles
    update <<< 256, 256 >>> (d_particles);


    // Unmap the SSBO to be available to OpenGL
    gpuErrchk(cudaGraphicsUnmapResources(1, &vboParticlesPos_cuda));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

RndTestSimGPU::RndTestSimGPU(Particles *particles, cudaGraphicsResource_t particlePos) : particles(particles),
                                                                                         vboParticlesPos_cuda(particlePos) {

}

