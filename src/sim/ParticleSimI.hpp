//
// Created by Pius Friesch on 19/11/2017.
//

#ifndef AGP_PROJECT_PARTICLECOMPUTATIONI_HPP
#define AGP_PROJECT_PARTICLECOMPUTATIONI_HPP

#include "Particle.hpp"
#include "../common.hpp"

class ParticleSimI {

public:
    virtual const std::vector<Particle *>& getParticles()=0;
    virtual void updateStep(int numTimeSteps)=0;
};

#endif //AGP_PROJECT_PARTICLECOMPUTATIONI_HPP
