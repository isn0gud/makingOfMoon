#include "RndTestSimCPU.hpp"

void RndTestSimCPU::updateStep(int numTimeSteps) {

    for (int i = 0; i < particles->numParticles; ++i) {
        particles->pos[i] += glm::vec4(0.01 * particles->pos[i].x, 0.01 * particles->pos[i].y,
                                       0.01 * particles->pos[i].z, 1);
    }
}

RndTestSimCPU::RndTestSimCPU(Particles *particles) : particles(particles) {}
