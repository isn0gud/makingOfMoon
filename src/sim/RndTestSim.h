#pragma once

#include "../common.hpp"
#include "Particles.h"
#include <vector>

class RndTestSim {

public: //TODO
    Particles *particles;

public:
    RndTestSim(Particles *particles);


    void step();

    /**
        * Generates a random particle position
        * @return 3D position + w component at 1.f
        */
    static glm::vec4 randomParticlePos();

    /**
     * Generates a random particle velocity
     * @param pos the same particle's position
     * @return 3D velocity + w component at 0.f
     */
    static glm::vec4 randomParticleVel(glm::vec4 pos);
};

