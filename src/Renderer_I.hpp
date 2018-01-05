#pragma once

#include "common.hpp"
#include <driver_types.h>
#include "Camera_I.hpp"
#include "InputHandler_I.hpp"

class Renderer_I {
public:

    virtual void init()=0;

    virtual glm::vec4 *allocateParticlesAndInit_cpu(int numParticles, glm::vec4 *particlesPos)=0;

    virtual cudaGraphicsResource_t allocateParticlesAndInit_gpu(int numParticles, glm::vec4 *particlesPos)=0;

    virtual void render()=0;

    virtual void destroy()=0;

    virtual Camera_I *getCamera()=0;
    virtual InputHandler_I *getInputHandler()=0;
};
