//
// Created by Pius Friesch on 19/11/2017.
//

#include "StaticVecFieldRndSim.hpp"

void StaticVecFieldRndSim::updateStep(int numTimeSteps) {

    for (Particle *p : particles) {
        p->velo = p->velo + p->accel * (float) numTimeSteps;
        p->pos = p->pos + p->velo * (float) numTimeSteps;
    }
}

#define POS_MIN 0.0f
#define POS_MAX (1.0f - POS_MIN)

#define VELO_MIN -0.00001f
#define VELO_MAX (0.00001f - VELO_MIN)

#define ACCEL_MIN -0.001f
#define ACCEL_MAX (0.001f - ACCEL_MIN)

#define PARTICLE_SIZE 0.05f


StaticVecFieldRndSim::StaticVecFieldRndSim() {
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        particles.push_back(new Particle(
                glm::vec3(((float) rand() / (float) (RAND_MAX)) * POS_MAX + POS_MIN,
                          ((float) rand() / (float) (RAND_MAX)) * POS_MAX + POS_MIN,
                          ((float) rand() / (float) (RAND_MAX)) * POS_MAX + POS_MIN),
                glm::vec3(((float) rand() / (float) (RAND_MAX)) * VELO_MAX + VELO_MIN,
                          ((float) rand() / (float) (RAND_MAX)) * VELO_MAX + VELO_MIN,
                          ((float) rand() / (float) (RAND_MAX)) * VELO_MAX + VELO_MIN),
                glm::vec3(((float) rand() / (float) (RAND_MAX)) * ACCEL_MAX + ACCEL_MIN,
                          ((float) rand() / (float) (RAND_MAX)) * ACCEL_MAX + ACCEL_MIN,
                          ((float) rand() / (float) (RAND_MAX)) * ACCEL_MAX + ACCEL_MIN),
                PARTICLE_SIZE));
    };
}

std::vector<Particle *> StaticVecFieldRndSim::getParticles() {
    return particles;
}

