//
// Created by Pius Friesch on 19/11/2017.
//

#ifndef ASS_OPENGL_STATICVECFIELDCOMPUTATION_HPP
#define ASS_OPENGL_STATICVECFIELDCOMPUTATION_HPP

#include "../common.hpp"
#include "Particle.hpp"
#include "ParticleSimI.hpp"

#define NUM_PARTICLES 100


class StaticVecFieldRndSim : public ParticleSimI {
private:

    std::vector<Particle *> particles;


public:
    StaticVecFieldRndSim();

    std::vector<Particle *> getParticles() override;

    void updateStep(int numTimeSteps, float stepSize) override;

};

#endif //ASS_OPENGL_STATICVECFIELDCOMPUTATION_HPP
