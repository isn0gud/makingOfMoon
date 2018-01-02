#pragma once


class Renderer_I {


public:

    virtual void init()=0;

    virtual glm::vec4 *allocateParticlesAndInit_cpu(int numParticles, glm::vec4 *particlesPos)=0;

    virtual cudaGraphicsResource_t allocateParticlesAndInit_gpu(int numParticles, glm::vec4 *particlesPos)=0;

    virtual void render()=0;

    virtual void destroy()=0;
};
