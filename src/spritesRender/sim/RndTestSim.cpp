//
// Created by friesch on 12/15/17.
//

#include "RndTestSim.h"
#include "Particles.h"

#include <random>

const float PI = 3.14159265358979323846;

using namespace std;

mt19937 rng;
uniform_real_distribution<> dis(0, 1);

glm::vec4 RndTestSim::randomParticlePos() {
    // Random position on a 'thick disk'
    glm::vec4 particle;
    float t = dis(rng) * 2 * PI;
    float s = dis(rng) * 100;
    particle.x = cos(t) * s;
    particle.y = sin(t) * s;
    particle.z = dis(rng) * 4;

    particle.w = 1.f;
    return particle;
}

glm::vec4 RndTestSim::randomParticleVel(glm::vec4 pos) {
    // Initial velocity is 'orbital' velocity from position
    glm::vec3 vel = glm::cross(glm::vec3(pos), glm::vec3(0, 0, 1));
    float orbital_vel = sqrt(2.0 * glm::length(vel));
    vel = glm::normalize(vel) * orbital_vel;
    return glm::vec4(vel, 0.0);
}

void RndTestSim::step() {

    for (int i = 0; i < particles->num_particles; ++i) {
        particles->particlePos[i] += glm::vec4(0.01 * particles->particlePos[i].x, 0.01 * particles->particlePos[i].y,
                                               0.01 * particles->particlePos[i].z, 1);
    }
    //TODO


}

RndTestSim::RndTestSim(Particles *particles) : particles(particles) {}
