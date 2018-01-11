#include <iostream>
#include <thread>

#include "WindowInputHandler.hpp"
#include "Timer.hpp"

#include "renderer/spritesRenderer/ParticleSpriteRenderer.cuh"
#include "renderer/spritesRenderer/SpriteRendererInputHandler.hpp"
#include "renderer/sphereRenderer/SphereRenderer.cuh"

#include "simulations/PlanetBuilder.hpp"
#include "simulations/gravitySim/GravitySimCPU.hpp"
#include "simulations/gravitySim/GravitySimGPU.cuh"

void displayOpenGLInfo() {
    // Display information about the GPU and OpenGL version
    printf("OpenGL %s\n", glewGetString(GLEW_VERSION));
    printf("Vendor: %s\n", glGetString(GL_VENDOR));
    printf("Renderer: %s\n", glGetString(GL_RENDERER));
    printf("Version: %s\n", glGetString(GL_VERSION));
    printf("GLSL: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
}

int main(int argc, char **argv) {

    int WINDOW_WIDTH = 800;
    int WINDOW_HEIGHT = 600;

    // Open window
    WindowManager *wm = WindowManager::getInstance();
    std::string windowTitle = "AGP Project - The Making of the Moon";
    wm->open(WINDOW_WIDTH, WINDOW_HEIGHT, windowTitle, true);

    // Create renderer
    Renderer_I *renderer = new SphereRenderer(WINDOW_WIDTH, WINDOW_HEIGHT);

    // Bind event listeners
    Camera_I *camera = renderer->getCamera();
    InputHandler_I *inputHandler = renderer->getInputHandler();
    WindowInputHandler windowInputHandler;
    wm->addKeyEventListener(&windowInputHandler);
    wm->addWindowEventListener(camera);
    wm->addKeyEventListener(inputHandler);
    wm->addScrollListener(inputHandler);
    wm->addCursorPositionListener(inputHandler);
    wm->addMouseButtonEventListener(inputHandler);

    renderer->init();

    // Build Scene
    float planet1fracton = 0.5;
    float planet2fracton = 0.5;
    int num_planet1 = (int) (planet1fracton * NUM_PARTICLES);
    int num_planet2 = (int) (planet2fracton * NUM_PARTICLES);
    assert(num_planet1 + num_planet2 == NUM_PARTICLES);

    Particles *particles = new Particles(NUM_PARTICLES);
    PlanetBuilder::buildPlanet(particles, 0, num_planet1,
                               Particles::TYPE::IRON, 3400.f,
                               Particles::TYPE::SILICATE, 6371.f,
                               glm::vec3(0), glm::vec3(0, 0, 0), glm::vec3(0, 7.2921159e-5, 0));
    PlanetBuilder::buildPlanet(particles, num_planet1, num_planet2,
                               Particles::TYPE::IRON, 3400.f,
                               Particles::TYPE::SILICATE, 6371.f,
                               glm::vec3(20000.0f, 0, 0), glm::vec3(-50, 0, -20), glm::vec3(0, 0, 0));

    // Init GPU Simulation and print GPU info
    GravitySimGPU sim(particles, renderer->allocateParticlesAndInit_gpu(particles));
    displayOpenGLInfo();

    // Start timer
    Timer timer;
    timer.start();

    // Main loop
    while (!wm->shouldClose()) {
        float frameTime = timer.getFrameTime();

        if (windowInputHandler.singleStepSimulation || windowInputHandler.runSimulation) {
            sim.updateStep(1);
            windowInputHandler.singleStepSimulation = false;
        }

        renderer->render(frameTime);
        wm->swapBuffers();
        wm->setTitle(windowTitle + " @" + std::to_string(1 / frameTime) + " fps");

    }

    // Clean up
    renderer->destroy();
    wm->close();
    return 0;
}
