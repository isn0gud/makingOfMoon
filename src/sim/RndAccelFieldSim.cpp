//
// Created by Pius Friesch on 20/11/2017.
//

#include "RndAccelFieldSim.hpp"
#include <cmath>

//    6.674×10−11 m3⋅kg−1⋅s−2
const float G = 6.67300E-11;


#define POS_MIN -1.0f
#define POS_MAX (1.0f - POS_MIN)

#define VELO_MIN -1E-4
#define VELO_MAX (1E-4*G - VELO_MIN)

#define ACCEL_MIN -G
#define ACCEL_MAX (G - ACCEL_MIN)

#define PARTICLE_SIZE 0.05f


void RndAccelFieldSim::updateStep(int numTimeSteps, float dt) {

    for (Particle *p : particles) {
        glm::vec3 total_grav = glm::vec3(0);

        //TODO: we should calc the updates in parallel and then update instead of on the elems directly:
        for (Particle *o : particles) {
            if (p != o) {
                // p.pos -> o.pos
                glm::vec3 dist_v = p->pos - o->pos;
                float dist = glm::length(dist_v);
                // p.pos -> o.pos w/ length 1.0
                glm::vec3 dir_v = (dist_v / dist);

                total_grav -= dir_v * (G * ((p->mass * o->mass) / (dist * dist)));

                if (dist <= (p->radius + o->radius)) {

                    //1. move them apart so they do not overlap
                    float overlap = (p->radius + o->radius) - dist;

                    p->pos = p->pos + ((GLfloat) 1.00001 * ((glm::length(p->pos) / dist) * overlap) * dir_v);
                    o->pos = o->pos + ((GLfloat) 1.00001 * ((glm::length(o->pos) / dist) * overlap) * -dir_v);

                    //update velocity - the velocity that is "consumed" by the impact
                    if (glm::length(p->velo) > 0 || glm::length(o->velo) > 0) {
                        glm::vec3 velo_norm = p->velo / glm::length(p->velo);
                        p->velo = (velo_norm - dir_v) * glm::length(p->velo);
                        velo_norm = o->velo / glm::length(o->velo);
                        o->velo = (velo_norm - dir_v) * glm::length(o->velo);
                    }
                }
            }
        }

        p->accel = total_grav;
        p->velo = p->velo + (p->accel * (GLfloat) numTimeSteps * SIM_SPEED);
        p->pos = p->pos + (p->velo * (GLfloat) numTimeSteps * SIM_SPEED);

    }
}


RndAccelFieldSim::RndAccelFieldSim() {
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        particles.push_back(new Particle(
                glm::vec3(((float) rand() / (float) (RAND_MAX)) * POS_MAX + POS_MIN,
                          ((float) rand() / (float) (RAND_MAX)) * POS_MAX + POS_MIN,
                          ((float) rand() / (float) (RAND_MAX)) * POS_MAX + POS_MIN),
                glm::vec3(0),
//                          ((float) rand() / (float) (RAND_MAX)) * VELO_MAX + VELO_MIN,
//                          ((float) rand() / (float) (RAND_MAX)) * VELO_MAX + VELO_MIN),
                glm::vec3(0),
//                glm::vec3(((float) rand() / (float) (RAND_MAX)) * ACCEL_MAX + ACCEL_MIN,
//                          ((float) rand() / (float) (RAND_MAX)) * ACCEL_MAX + ACCEL_MIN,
//                          ((float) rand() / (float) (RAND_MAX)) * ACCEL_MAX + ACCEL_MIN),
                PARTICLE_SIZE));
    };
}

std::vector<Particle *> RndAccelFieldSim::getParticles() {
    return particles;
}

