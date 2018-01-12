#include <iostream>
#include <thread>

#include "WindowInputHandler.hpp"
#include "Timer.hpp"

#include "renderer/spritesRenderer/ParticleSpriteRenderer.hpp"
#include "renderer/spritesRenderer/SpriteRendererInputHandler.hpp"
#include "renderer/sphereRenderer/SphereRenderer.cuh"

#include "simulations/PlanetBuilder.hpp"
#include "simulations/gravitySim/GravitySimCPU.hpp"
#include "simulations/gravitySim/GravitySimGPU.hpp"

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
//    Renderer_I *renderer = new SphereRenderer(WINDOW_WIDTH, WINDOW_HEIGHT);
    Renderer_I *renderer = new ParticleSpriteRenderer(WINDOW_WIDTH, WINDOW_HEIGHT);

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
    int NUM_PARTICLESs = 1000;

    // Build Scene
    float planet1fracton = 0.5;
    float planet2fracton = 0.5;
    int num_planet1 = (int) (planet1fracton * NUM_PARTICLESs);
    int num_planet2 = (int) (planet2fracton * NUM_PARTICLESs);
    assert(num_planet1 + num_planet2 == NUM_PARTICLESs);

    Particles::ParticlesInit pInit;

    Particles::ParticlesInit pInit1 = PlanetBuilder::buildPlanet(
           num_planet1,
            TYPE::IRON, 3400.f,
            TYPE::SILICATE, 6371.f,
            //glm::vec3(0), glm::vec3(0), glm::vec3(0, 7.2921159e-5, 0),
            glm::vec3(0), glm::vec3(0, 0, 0), glm::vec3(0, 0, 0));
    Particles::ParticlesInit pInit2 = PlanetBuilder::buildPlanet(
            num_planet2,
            TYPE::IRON, 3400.f,
            TYPE::SILICATE, 6371.f,
            //glm::vec3(0), glm::vec3(0), glm::vec3(0, 7.2921159e-5, 0),
            glm::vec3(10000.0f), glm::vec3(0, 0, 0),
            glm::vec3(0, 0, 0));

    pInit.addParticles(pInit1);
    pInit.addParticles(pInit2);

    Particles *particles = new Particles(pInit);


    // Init GPU Simulation and print GPU info
//    GravitySimGPU sim(particles, renderer->allocateParticlesAndInit_gpu(particles));

    /* CPU */
    particles->setParticlePos(renderer->allocateParticlesAndInit_cpu(particles));
    GravitySimCPU sim(particles);


    displayOpenGLInfo();

    // Start timer
    Timer timer;
    timer.start();

    double fpsAverage = 1;
    double smoothing = 0.995;

    // Main loop
    while (!wm->shouldClose()) {

        float frameTime = timer.getFrameTime();

        if (windowInputHandler.singleStepSimulation || windowInputHandler.runSimulation) {
            sim.updateStep(1);
            windowInputHandler.singleStepSimulation = false;
            fpsAverage = (fpsAverage * smoothing) + (((double) 1 / frameTime) * (1.0 - smoothing));
        }

        if (frameTime < 0.001)
            usleep(900);

        renderer->render(frameTime);
        wm->swapBuffers();
        wm->setTitle(windowTitle + " @" + std::to_string(1 / frameTime) + " fps, " + "Average: " +
                     std::to_string(fpsAverage));

    }

    // Clean up
    renderer->destroy();
    wm->close();
    return 0;
}
