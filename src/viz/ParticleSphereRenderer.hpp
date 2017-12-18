#ifndef AGP_PROJECT_RENDERER_HPP
#define AGP_PROJECT_RENDERER_HPP

#include "../common.hpp"
#include "../util/WindowManager.hpp"

#include "Camera.hpp"
#include "Sphere.hpp"
#include "Model.hpp"
#include "RendererInputHandler.hpp"
#include "../util/ShaderProgram.hpp"

class ParticleSphereRenderer : public WindowEventListener
{
private:
    ShaderProgram programmMain;

protected:
    Camera camera;
//    GLint shaderProgramId;
    Model sphereModel;

    float cameraAzimuthAngle;
    float cameraPolarAngle;

    RendererInputHandler inputHandler;

    void updateCamera(float frameTime);

public:

    void init(int windowWidth, int windowHeight);
    void render(const std::vector<Sphere*>& spheres, float frameTime);
//    void clear();

    void onWindowSizeChanged(int width, int height);
};

#endif
