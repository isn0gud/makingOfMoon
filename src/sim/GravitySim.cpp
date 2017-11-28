//
// Created by Karl Kvarnfors on 26/11/2017.
//

#include "GravitySim.hpp"
#include "LatticeBuilder.hpp"
#include <cmath>

#define CUBE_SIDE 5
#define SIM_SPEED 3
#define COLL_SPEED 1.5

//    6.674×10−11 m3⋅kg−1⋅s−2
const float G = 6.67300E-11;
const float particleRadius = 0.1;
const float distanceEpsilon = 0.05;
const float particleMass = 1e7;


void GravitySim::updateStep(int numTimeSteps, float dt)
{
    if(forces.size() != particles.size())
        forces.resize(particles.size());

    for(int i = 0; i < forces.size(); i++)
        forces[i] = glm::vec3(0,0,0);

    // Iterate using indices (yes, it's ugly but it's faster)
    for(int i = 0; i < particles.size(); i++)
    {
        for(int j = i+1; j < particles.size(); j++)
        {
            glm::vec3 force(0,0,0);
            glm::vec3 difference = particles[i]->pos - particles[j]->pos;
            float distance = glm::length(difference);
            glm::vec3 differenceNormal = (difference / distance);

            // Prevent numerical errors from divide by zero
            if(distance < distanceEpsilon)
                distance = distanceEpsilon;

            // Newtonian gravity (F_g = -G m_1 m_2 r^(-2) hat{n})
            force -= differenceNormal * (G * ((particles[i]->mass * particles[j]->mass) / (distance * distance)));

            // Separation "spring" (F_s = k_eff (r_1+r_2 - r))
            if(distance < particles[i]->radius + particles[j]->radius)
            {
                float elasticConstants[2];
                elasticConstants[0] = particles[i]->elasticSpringConstant;
                elasticConstants[1] = particles[j]->elasticSpringConstant;

                // If the separation increases
                if(dot(differenceNormal, particles[i]->velo - particles[j]->velo) > 0)
                {
                    if(distance < particles[i]->radius*particles[i]->shellDepthFraction + particles[j]->radius*particles[j]->shellDepthFraction)
                    {
                        elasticConstants[0] *= particles[i]->inelasticSpringForceReductionFactor;
                        elasticConstants[1] *= particles[j]->inelasticSpringForceReductionFactor;
                    }
                    else if(distance < particles[i]->radius*particles[i]->shellDepthFraction + particles[j]->radius)
                        elasticConstants[0] *= particles[i]->inelasticSpringForceReductionFactor;
                    else if(distance < particles[i]->radius + particles[j]->radius*particles[j]->shellDepthFraction)
                        elasticConstants[1] *= particles[j]->inelasticSpringForceReductionFactor;
                }

                float efficientSpringConstant = (elasticConstants[0] + elasticConstants[1]);
                force += differenceNormal * efficientSpringConstant * ((particles[i]->radius + particles[j]->radius)*(particles[i]->radius + particles[j]->radius) - distance*distance);
            }

            forces[i] += force;
            forces[j] -= force;
        }
    }

    // Euler integration
    for(int i = 0; i < forces.size(); i++)
    {
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
                particles.push_back(new Particle(
                    glm::vec3(
                        ((float)x-((float)(CUBE_SIDE-1)/2.0f))/5.f,
                        ((float)y-((float)(CUBE_SIDE-1)/2.0f))/5.f,
                        ((float)z-((float)(CUBE_SIDE-1)/2.0f))/5.f),
                    glm::vec3(0),
                    glm::vec3(0),
                    particleRadius,
                    particleMass));
                particles[index]->velo += glm::cross(glm::vec3(0,0,0.25), particles[index]->pos);
                particles[index]->shellDepthFraction = 0.9;
                particles[index]->elasticSpringConstant = particleMass*25;
                particles[index]->inelasticSpringForceReductionFactor = 0.01;
                index++;
            }
        }
    }

    // Build planet at to the right

    for (int x = 0; x < CUBE_SIDE; x++) {
        for (int y = 0; y < CUBE_SIDE; y++) {
            for (int z = 0; z < CUBE_SIDE; z++) {
                particles.push_back(new Particle(
                    glm::vec3(
                        ((float)x-((float)(CUBE_SIDE-1)/2.0f))/3.f + 35,
                        ((float)y-((float)(CUBE_SIDE-1)/2.0f))/3.f,
                        ((float)z-((float)(CUBE_SIDE-1)/2.0f))/3.f),
                    glm::vec3(-1*COLL_SPEED,0.01*COLL_SPEED,0),
                    glm::vec3(0),
                    particleRadius,
                    particleMass));
                particles[index]->velo += glm::cross(glm::vec3(0,0,-0.05), particles[index]->pos - glm::vec3(35, 0, 0));
                particles[index]->shellDepthFraction = 0.9;
                particles[index]->elasticSpringConstant = particleMass*10;
                particles[index]->inelasticSpringForceReductionFactor = 0.01;
                index++;
            }
        }
    }
}


std::vector<Particle *> GravitySim::getParticles() {
    return particles;
}

