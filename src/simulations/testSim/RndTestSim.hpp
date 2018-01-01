#pragma once

#include "../../common.hpp"
#include "../../Particles.hpp"
#include "../../ParticleSimI.hpp"
#include <vector>
#include "../gravitySim/PlanetBuilder.hpp"

glm::vec4 randomParticlePos();

class RndTestSim : public ParticleSimI {
private:


    Particles *particles = nullptr;

public:

    /**
     *
     * @param particles Particles struct initialized by the renderer.
     */
    void initParticles(Particles *p) override;

    void updateStep(int numTimeSteps) override;


};

