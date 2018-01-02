#pragma once

#include "../../common.hpp"
#include "../../Renderer_I.hpp"
#include "../spritesRenderer/CameraRotateCenter.hpp"
#include "../../ShaderProgram.hpp"


//#include "Camera.hpp"
#include "Sphere.hpp"
#include "Model.hpp"

class Renderer : public RendererCPU {
protected:


//    Camera camera;
    ShaderProgram shaderProgram;
    Model sphereModel;

    float cameraAzimuthAngle;
    float cameraPolarAngle;

    CameraRotateCenter *camera;

    void updateCamera(float frameTime);

public:
    void init() override;

    Particles *allocateParticles(int numParticles) override;

    void render() override;

    void destroy() override;

    Renderer(CameraRotateCenter *camera);

//    void init(int windowWidth, int windowHeight);

//    void render(const std::vector<Sphere *> &spheres, float frameTime);
//
//    void clear();
//
//    void onWindowSizeChanged(int width, int height);
};