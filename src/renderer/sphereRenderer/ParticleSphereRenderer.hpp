#pragma once

#include <driver_types.h>
#include "../../common.hpp"
#include "../../Renderer_I.hpp"
#include "../spritesRenderer/CameraRotateCenter.hpp"
#include "../../ShaderProgram.hpp"
#include "../../Camera_I.hpp"

#include "Sphere.hpp"
#include "Model.hpp"

class ParticleSphereRenderer : public Renderer_I {
protected:


//    Camera camera;
    ShaderProgram shaderProgram;
    Model sphereModel;

    float cameraAzimuthAngle;
    float cameraPolarAngle;

    Camera_I *camera;

    void updateCamera(float frameTime);

public:
    void init() override;

    glm::vec4 *allocateParticlesAndInit_cpu(int numParticles, glm::vec4 *particlesPos) override;

    cudaGraphicsResource_t allocateParticlesAndInit_gpu(int numParticles, glm::vec4 *particlesPos) override;

    void render() override;

    void destroy() override;

//    void init() override;
//
//    Particles *allocateParticles(int numParticles) override;
//
//    void render() override;
//
//    void destroy() override;
//
//    ParticleSphereRenderer(CameraRotateCenter *camera);

//    void init(int windowWidth, int windowHeight);

//    void render(const std::vector<Sphere *> &spheres, float frameTime);
//
//    void clear();
//
//    void onWindowSizeChanged(int width, int height);
};