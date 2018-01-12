#include "GravitySimCPU.hpp"
#include <cmath>

#define timeStep 10.0f
#define COLL_SPEED 1.5
#define CUBE_SIDE 5

//  units: SI, but km instead of m
//  6.674×10−20 (km)^3⋅kg^(−1)⋅s^(−2)
const float G = 6.674E-20;
const float distanceEpsilon = 47.0975;

static struct ParticleConst {
    float elasticSpringConstant;
    float shellDepthFraction;
    float inelasticSpringForceReductionFactor;
} ironConst = {.elasticSpringConstant = IRON_elasticSpringConstant,
        .shellDepthFraction = IRON_shellDepthFraction,
        .inelasticSpringForceReductionFactor = IRON_inelasticSpringForceReductionFactor},
        silicateConst = {.elasticSpringConstant = SILICATE_elasticSpringConstant,
        .shellDepthFraction = SILICATE_shellDepthFraction,
        .inelasticSpringForceReductionFactor = SILICATE_inelasticSpringForceReductionFactor};


void GravitySimCPU::updateStep(int numTimeSteps) {


    for (int step = 0; step < numTimeSteps; step++) {

        std::vector<glm::vec3> forces;
        if (forces.size() != particles->numParticles)
            forces.resize(particles->numParticles);

        // Iterate using indices (yes, it's ugly but it's faster)
        for (int i = 0; i < particles->numParticles; i++) {
            for (int j = i + 1; j < particles->numParticles; j++) {
                glm::vec3 force(0, 0, 0);
                glm::vec3 difference = glm::vec3(particles->pos__radius[i]) - glm::vec3(particles->pos__radius[j]);

                float distance = glm::length(difference);
                glm::vec3 differenceNormal = (difference / distance);

                // Prevent numerical errors from divide by zero
                if (distance < distanceEpsilon) {
                    //printf("The repulsive parameters were not set strong enough!\n");
                    distance = distanceEpsilon;
                }

                // Newtonian gravity (F_g = -G m_1 m_2 r^(-2) hat{n}), doubles are needed to prevent overflow, needs to be fixed in GPU implementation
                force -= differenceNormal * (float) ((double) G *
                                                     (((double) particles->velo__mass[i].w *
                                                       (double) particles->velo__mass[j].w) /
                                                      ((double) (distance * distance))));

                // Separation "spring"
                if (distance < particles->pos__radius[i].w + particles->pos__radius[j].w) {
                    ParticleConst pConst1 = (particles->type[i] == TYPE::IRON) ? ironConst : silicateConst;
                    ParticleConst pConst2 = (particles->type[j] == TYPE::IRON) ? ironConst : silicateConst;

                    float elasticConstantParticle1 = pConst1.elasticSpringConstant;
                    float elasticConstantParticle2 = pConst2.elasticSpringConstant;

                    // If the separation increases, i.e. the separation velocity is positive
                    if (dot(differenceNormal, glm::vec3(particles->velo__mass[i])
                                              - glm::vec3(particles->velo__mass[j])) > 0) {
                        // Check if the force shall be reduced due to plastic deformation
                        if (distance < particles->pos__radius[i].w * pConst1.shellDepthFraction +
                                       particles->pos__radius[j].w * pConst2.shellDepthFraction) {
                            //case 1d
                            elasticConstantParticle1 *= pConst1.inelasticSpringForceReductionFactor;
                            elasticConstantParticle2 *= pConst2.inelasticSpringForceReductionFactor;
                        } else if (distance <
                                   particles->pos__radius[i].w * pConst1.shellDepthFraction +
                                   particles->pos__radius[j].w) {
                            // case 1c 1
                            elasticConstantParticle1 *= pConst1.inelasticSpringForceReductionFactor;
                        } else if (distance <
                                   particles->pos__radius[i].w +
                                   particles->pos__radius[j].w * pConst2.shellDepthFraction) {
                            //case 1c 2
                            elasticConstantParticle2 *= pConst2.inelasticSpringForceReductionFactor;
                        }
                    }

                    float efficientSpringConstant = (elasticConstantParticle1 + elasticConstantParticle2);
                    // Add compression force (F_s = k_eff ((r_1+r_2)^2 - r^2))
                    force += differenceNormal * efficientSpringConstant *
                             ((particles->pos__radius[i].w + particles->pos__radius[j].w) *
                              (particles->pos__radius[i].w + particles->pos__radius[j].w) -
                              distance * distance);
                }

                // Newton's third law
                forces[i] += force;
                forces[j] -= force;
            }
        }

        // Leapfrog integration (better than Euler for gravity simulations)
        for (int i = 0; i < forces.size(); i++) {
            glm::vec3 newAcceleration = forces[i] / particles->velo__mass[i].w; // a_i+1 = F_i+1 / m
            particles->pos__radius[i] += glm::vec4(glm::vec3(particles->velo__mass[i]) * timeStep +
                                                   glm::vec3(particles->accel[i]) * 0.5f * timeStep *
                                                   timeStep, 0); // x_i+1 = v_i*dt + a_i*dt^2/2
            particles->velo__mass[i] +=
                    glm::vec4((glm::vec3(particles->accel[i]) + newAcceleration) * 0.5f * timeStep, 0);
            // v_i+1 = v_i + (a_i + a_i+1)dt/2
            particles->accel[i] = glm::vec4(newAcceleration, 1);
        }
    }
}

GravitySimCPU::~GravitySimCPU() {
    // deletes all particles
    particles->clear();
}

GravitySimCPU::GravitySimCPU(Particles *particles) : particles(particles) {}
