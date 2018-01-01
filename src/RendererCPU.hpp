#pragma once

#include "Particles.hpp"
#include "Camera.hpp"

class RendererCPU {


public:

    virtual void init()=0;

    virtual Particles *allocateParticles(int numParticles)=0;


    virtual void render()=0;

    virtual void destroy()=0;
};
