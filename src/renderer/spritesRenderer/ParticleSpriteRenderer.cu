#include "ParticleSpriteRenderer.hpp"

#include <algorithm>
#include <glm/gtc/type_ptr.hpp>
#include <cuda_gl_interop.h>
#include "CameraRotateCenter.hpp"


cudaGraphicsResource_t ParticleSpriteRenderer::allocateParticlesAndInit_gpu(Particles* particles) {
    // SSBO allocation & data upload
    glNamedBufferStorage(vboParticlesPos, particles->numParticles * sizeof(glm::vec4), particles->pos__radius,
                         GL_MAP_WRITE_BIT | GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT |
                         GL_MAP_COHERENT_BIT); // Buffer storage is fixed size compared to BuferData
    this->numParticles = static_cast<size_t>(particles->numParticles);

    cudaGraphicsResource_t vboParticlesPos_cuda;
    cudaGraphicsGLRegisterBuffer(&vboParticlesPos_cuda,
                                 vboParticlesPos,
                                 cudaGraphicsRegisterFlagsNone);
    return vboParticlesPos_cuda;
}
