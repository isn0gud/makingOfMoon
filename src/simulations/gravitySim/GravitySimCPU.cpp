#include "GravitySimCPU.hpp"
#include <cmath>

#define timeStep 10.0f
#define COLL_SPEED 1.5
#define CUBE_SIDE 5

//  units: SI, but km instead of m
//  6.674×10−20 (km)^3⋅kg^(−1)⋅s^(−2)
const float G = 6.674E-20;
const float distanceEpsilon = 47.0975;

void GravitySimCPU::updateStep(int numTimeSteps) {
    if (forces.size() != particles->numParticles)
        forces.resize(particles->numParticles);

    for (int step = 0; step < numTimeSteps; step++) {

        for (auto &force : forces)
            force = glm::vec3(0);

        // Iterate using indices (yes, it's ugly but it's faster)
        for (int i = 0; i < particles->numParticles; i++) {
            for (int j = i + 1; j < particles->numParticles; j++) {
                glm::vec3 force(0, 0, 0);
                glm::vec3 difference = glm::vec3(particles->pos[i]) - glm::vec3(particles->pos[j]);

                float distance = glm::length(difference);
                glm::vec3 differenceNormal = (difference / distance);

                // Prevent numerical errors from divide by zero
                if (distance < distanceEpsilon) {
                    //printf("The repulsive parameters were not set strong enough!\n");
                    distance = distanceEpsilon;
                }

                // Newtonian gravity (F_g = -G m_1 m_2 r^(-2) hat{n}), doubles are needed to prevent overflow, needs to be fixed in GPU implementation
                force -= differenceNormal * (float) ((double) G *
                                                     (((double) particles->mass[i] * (double) particles->mass[j]) /
                                                      ((double) (distance * distance))));

                // Separation "spring"
                if (distance < particles->radius[i] + particles->radius[j]) {
                    float elasticConstantParticle1 = particles->elasticSpringConstant[i];
                    float elasticConstantParticle2 = particles->elasticSpringConstant[j];

                    // If the separation increases, i.e. the separation velocity is positive
                    if (dot(differenceNormal, glm::vec3(particles->velo[i] - particles->velo[j])) > 0) {
                        // Check if the force shall be reduced due to plastic deformation
                        if (distance < particles->radius[i] * particles->shellDepthFraction[i] +
                                       particles->radius[j] * particles->shellDepthFraction[j]) {
                            //case 1d
                            elasticConstantParticle1 *= particles->inelasticSpringForceReductionFactor[i];
                            elasticConstantParticle2 *= particles->inelasticSpringForceReductionFactor[j];
                        } else if (distance <
                                   particles->radius[i] * particles->shellDepthFraction[i] + particles->radius[j]) {
                            // case 1c 1
                            elasticConstantParticle1 *= particles->inelasticSpringForceReductionFactor[i];
                        } else if (distance <
                                   particles->radius[i] + particles->radius[j] * particles->shellDepthFraction[j]) {
                            //case 1c 2
                            elasticConstantParticle2 *= particles->inelasticSpringForceReductionFactor[j];
                        }
                    }

                    float efficientSpringConstant = (elasticConstantParticle1 + elasticConstantParticle2);
                    // Add compression force (F_s = k_eff ((r_1+r_2)^2 - r^2))
                    force += differenceNormal * efficientSpringConstant *
                             ((particles->radius[i] + particles->radius[j]) *
                              (particles->radius[i] + particles->radius[j]) -
                              distance * distance);
                }

                // Newton's third law
                forces[i] += force;
                forces[j] -= force;
            }
        }

        // Leapfrog integration (better than Euler for gravity simulations)
        for (int i = 0; i < forces.size(); i++) {
            glm::vec4 newAcceleration = glm::vec4(forces[i] / particles->mass[i], 1); // a_i+1 = F_i+1 / m
            particles->pos[i] += particles->velo[i] * timeStep +
                                 particles->accel[i] * 0.5f * timeStep *
                                 timeStep; // x_i+1 = v_i*dt + a_i*dt^2/2
            particles->velo[i] +=
                    (particles->accel[i] + newAcceleration) * 0.5f * timeStep; // v_i+1 = v_i + (a_i + a_i+1)dt/2
            particles->accel[i] = newAcceleration;
            particles->pos[i].w = 1;
        }
    }
}

GravitySimCPU::~GravitySimCPU() {
    // deletes all particles
    particles->clear();
}

GravitySimCPU::GravitySimCPU(Particles *particles) : particles(particles) {}
