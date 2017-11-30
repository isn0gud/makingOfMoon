//
// Created by Karl Kvarnfors on 26/11/2017.
//

#include "GravitySim.hpp"
#include <cmath>

#define CUBE_SIDE 5
#define SIM_SPEED 3
#define COLL_SPEED 1.5

//    6.674×10−11 m3⋅kg−1⋅s−2
const float G = 6.674E-11;
//unit: km
//const float distanceEpsilon = 47.0975;
const float distanceEpsilon = 0.05;
const float particleMass = 1e7;


void GravitySim::updateStep(int numTimeSteps, float dt) {
    if (forces.size() != particles.size())
        forces.resize(particles.size());

    for (int i = 0; i < forces.size(); i++)
        forces[i] = glm::vec3(0);

    // Iterate using indices (yes, it's ugly but it's faster)
    for (int i = 0; i < particles.size(); i++) {
        for (int j = i + 1; j < particles.size(); j++) {
            glm::vec3 force(0, 0, 0);
            glm::vec3 difference = particles[i]->pos - particles[j]->pos;
            float distance = glm::length(difference);
            glm::vec3 differenceNormal = (difference / distance);

            // Prevent numerical errors from divide by zero
            if (distance < distanceEpsilon) {
                printf("The repulsive parameters were not set strong enough!\n");
                distance = distanceEpsilon;
            }

            // Newtonian gravity (F_g = -G m_1 m_2 r^(-2) hat{n})
            force -= differenceNormal * (G * ((particles[i]->mass * particles[j]->mass) / (distance * distance)));

            // Separation "spring" (F_s = k_eff (r_1+r_2 - r))
            if (distance < particles[i]->radius + particles[j]->radius) {
                float elasticConstantParticle1 = particles[i]->elasticSpringConstant;
                float elasticConstantParticle2 = particles[j]->elasticSpringConstant;

                // If the separation increases
                if (dot(differenceNormal, particles[i]->velo - particles[j]->velo) > 0) {
                    //the particle do not move in the same direction //TODO what about different speeds?
                    if (distance < particles[i]->radius * particles[i]->shellDepthFraction +
                                   particles[j]->radius * particles[j]->shellDepthFraction) {
                        //case 1d
                        elasticConstantParticle1 *= particles[i]->inelasticSpringForceReductionFactor;
                        elasticConstantParticle2 *= particles[j]->inelasticSpringForceReductionFactor;
                    } else if (distance <
                               particles[i]->radius * particles[i]->shellDepthFraction +
                               particles[j]->radius) {
                        // case 1c 1
                        elasticConstantParticle1 *= particles[i]->inelasticSpringForceReductionFactor;
                    } else if (distance <
                               particles[i]->radius + particles[j]->radius * particles[j]->shellDepthFraction) {
                        //case 1c 2
                        elasticConstantParticle2 *= particles[j]->inelasticSpringForceReductionFactor;
                    }
                }

                float efficientSpringConstant = (elasticConstantParticle1 + elasticConstantParticle2);
                // "reducing" the force of gravity so the particles don't implode
                force += differenceNormal * efficientSpringConstant *
                         ((particles[i]->radius + particles[j]->radius) *
                          (particles[i]->radius + particles[j]->radius) -
                          distance * distance);
            }

            forces[i] += force;
            forces[j] -= force;
        }
    }

    // Euler integration //TODO error prone??
    for (int i = 0; i < forces.size(); i++) {
        particles[i]->accel = forces[i] / particles[i]->mass;
        particles[i]->pos += particles[i]->velo * numTimeSteps * SIM_SPEED * dt;
        particles[i]->velo += particles[i]->accel * numTimeSteps * SIM_SPEED * dt;
    }
}

GravitySim::GravitySim() {

    // TODO: Make class for generating inital states, change to fcc-lattice for a more realistic inital state

    // Build planet at origin
    int index = 0;
    for (int x = 0; x < CUBE_SIDE; x++) {
        for (int y = 0; y < CUBE_SIDE; y++) {
            for (int z = 0; z < CUBE_SIDE; z++) {
                particles.push_back(Particle::ironParticle(glm::vec3(
                        ((float) x - ((float) (CUBE_SIDE - 1) / 2.0f)) / 3.f,
                        ((float) y - ((float) (CUBE_SIDE - 1) / 2.0f)) / 3.f,
                        ((float) z - ((float) (CUBE_SIDE - 1) / 2.0f))  / 3.f)));
                particles[index]->velo += glm::cross(glm::vec3(0, 0, 0.25), particles[index]->pos);
                index++;
            }
        }
    }

    // Build planet at to the right

//    for (int x = 0; x < CUBE_SIDE; x++) {
//        for (int y = 0; y < CUBE_SIDE; y++) {
//            for (int z = 0; z < CUBE_SIDE; z++) {
//                particles.push_back(Particle::silicateParticle(
//                        glm::vec3(
//                                ((float) x - ((float) (CUBE_SIDE - 1) / 2.0f)) / 5.f + 3,
//                                ((float) y - ((float) (CUBE_SIDE - 1) / 2.0f)) / 5.f,
//                                ((float) z - ((float) (CUBE_SIDE - 1) / 2.0f)) / 5.f)));
//                particles[index]->velo += glm::cross(glm::vec3(0, 0, -0.05),
//                                                     particles[index]->pos - glm::vec3(3, 0, 0));
//                index++;
//            }
//        }
//    }
}


std::vector<Particle *> GravitySim::getParticles() {
    return particles;
}

GravitySim::~GravitySim() {
    // deletes all particles
    particles.clear();
}

