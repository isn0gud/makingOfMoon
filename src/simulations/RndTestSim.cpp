#include "RndTestSim.hpp"

#include <random>

const float PI = 3.14159265358979323846f;

using namespace std;

mt19937 rng;
uniform_real_distribution<> dis(0, 1);


/**
    * Generates a random particle position
    * @return 3D position + w component at 1.f
    */
glm::vec4 randomParticlePos() {
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

/**
 * Generates a random particle velocity
 * @param pos the same particle's position
 * @return 3D velocity + w component at 0.f
 */
glm::vec4 randomParticleVel(glm::vec4 pos) {
    // Initial velocity is 'orbital' velocity from position
    glm::vec3 vel = glm::cross(glm::vec3(pos), glm::vec3(0, 0, 1));
    float orbital_vel = sqrt(2.0 * glm::length(vel));
    vel = glm::normalize(vel) * orbital_vel;
    return glm::vec4(vel, 0.0);
}


void RndTestSim::updateStep(int numTimeSteps) {

    for (int i = 0; i < particles->numParticles; ++i) {
        particles->particlePos[i] += glm::vec4(0.01 * particles->particlePos[i].x, 0.01 * particles->particlePos[i].y,
                                               0.01 * particles->particlePos[i].z, 1);
    }
}

void RndTestSim::initParticles(Particles *p) {
    particles = p;

    for (size_t i = 0; i < particles->numParticles; ++i) {
        particles->particlePos[i] = randomParticlePos();
        particles->particleVelo[i] = randomParticleVel(particles->particlePos[i]);
    }
}
