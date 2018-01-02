#pragma once

#include "../../common.hpp"
#include "../../Particles.hpp"
#include "../../ParticleSim_I.hpp"
#include <vector>
#include "../PlanetBuilder.hpp"


class RndTestSimCPU : public ParticleSim_I {
private:
public:
    RndTestSimCPU(Particles *particles);

private:

    Particles *particles = nullptr;

public:

    void updateStep(int numTimeSteps) override;


};

