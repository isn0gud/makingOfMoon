//
// Created by Karl Kvarnfors on 26/11/2017.
//

#include "GravitySim.hpp"
#include "PlanetBuilder.hpp"
#include <cmath>

#define timeStep 10.0f
#define COLL_SPEED 1.5
#define CUBE_SIDE 5

//  units: SI, but km instead of m
//  6.674×10−20 (km)^3⋅kg^(−1)⋅s^(−2)
const float G = 6.674E-20;
const float distanceEpsilon = 47.0975;

void GravitySim::updateStep(int numTimeSteps) {
    for(int step = 0; step < numTimeSteps; step++)
    {
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
                    //printf("The repulsive parameters were not set strong enough!\n");
                    distance = distanceEpsilon;
                }

                // Newtonian gravity (F_g = -G m_1 m_2 r^(-2) hat{n}), doubles are needed to prevent overflow, needs to be fixed in GPU implementation
                force -= differenceNormal * (float)((double)G * (((double)particles[i]->mass * (double)particles[j]->mass) / ((double)(distance * distance))));

                // Separation "spring"
                if (distance < particles[i]->radius + particles[j]->radius) {
                    float elasticConstantParticle1 = particles[i]->elasticSpringConstant;
                    float elasticConstantParticle2 = particles[j]->elasticSpringConstant;

                    // If the separation increases, i.e. the separation velocity is positive
                    if (dot(differenceNormal, particles[i]->velo - particles[j]->velo) > 0) {
                        // Check if the force shall be reduced due to plastic deformation
                        if (distance < particles[i]->radius * particles[i]->shellDepthFraction +
                                       particles[j]->radius * particles[j]->shellDepthFraction) {
                            //case 1d
                            elasticConstantParticle1 *= particles[i]->inelasticSpringForceReductionFactor;
                            elasticConstantParticle2 *= particles[j]->inelasticSpringForceReductionFactor;
                        } else if (distance <
                                   particles[i]->radius * particles[i]->shellDepthFraction + particles[j]->radius) {
                            // case 1c 1
                            elasticConstantParticle1 *= particles[i]->inelasticSpringForceReductionFactor;
                        } else if (distance <
                                   particles[i]->radius + particles[j]->radius * particles[j]->shellDepthFraction) {
                            //case 1c 2
                            elasticConstantParticle2 *= particles[j]->inelasticSpringForceReductionFactor;
                        }
                    }

                    float efficientSpringConstant = (elasticConstantParticle1 + elasticConstantParticle2);
                    // Add compression force (F_s = k_eff ((r_1+r_2)^2 - r^2))
                    force += differenceNormal * efficientSpringConstant *
                             ((particles[i]->radius + particles[j]->radius) *
                              (particles[i]->radius + particles[j]->radius) -
                              distance * distance);
                }

                // Newton's third law
                forces[i] += force;
                forces[j] -= force;
            }
        }

        // Leapfrog integration (better than Euler for gravity simulations)
        for (int i = 0; i < forces.size(); i++)
        {
            glm::vec3 newAcceleration = forces[i] / particles[i]->mass; // a_i+1 = F_i+1 / m
            particles[i]->pos += particles[i]->velo * timeStep + particles[i]->accel* 0.5 * timeStep * timeStep; // x_i+1 = v_i*dt + a_i*dt^2/2
            particles[i]->velo += (particles[i]->accel + newAcceleration) * 0.5 * timeStep; // v_i+1 = v_i + (a_i + a_i+1)dt/2
            particles[i]->accel = newAcceleration;
        }
    }
}

GravitySim::GravitySim()
{
    PlanetBuilder::buildPlanet(500,
                               Particle::TYPE::IRON, 1220.f*0.25f,
                               Particle::TYPE::SILICATE, 6371.f*0.25f,
                               //glm::vec3(0), glm::vec3(0), glm::vec3(0, 7.2921159e-5, 0),
                               glm::vec3(0), glm::vec3(0), glm::vec3(0, 0, 0),
                               particles);

    // Build planet at origin

//    int index = 0;
//    for (int x = 0; x < CUBE_SIDE; x++) {
//        for (int y = 0; y < CUBE_SIDE; y++) {
//            for (int z = 0; z < CUBE_SIDE; z++) {
//                particles.push_back(Particle::ironParticle(glm::vec3(
//                        ((float) x - ((float) (CUBE_SIDE - 1) / 2.0f)) / 3.f,
//                        ((float) y - ((float) (CUBE_SIDE - 1) / 2.0f)) / 3.f,
//                        ((float) z - ((float) (CUBE_SIDE - 1) / 2.0f))  / 3.f)));
//                particles[index]->velo += glm::cross(glm::vec3(0, 0, 0.25), particles[index]->pos);
//                index++;
//            }
//        }
//    }

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


const std::vector<Particle *>& GravitySim::getParticles() {
    return particles;
}

GravitySim::~GravitySim() {
    // deletes all particles
    particles.clear();
}

