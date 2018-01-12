#include "RndTestSimCPU.hpp"

void RndTestSimCPU::updateStep(int numTimeSteps) {

    for (int i = 0; i < particles->numParticles; ++i) {
        particles->pos__radius[i] += glm::vec4(0.01 * particles->pos__radius[i].x, 0.01 * particles->pos__radius[i].y,
                                       0.01 * particles->pos__radius[i].z, 1);
    }
}

RndTestSimCPU::RndTestSimCPU(Particles *particles) : particles(particles) {}
