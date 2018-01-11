#include <iostream>

#include "WindowInputHandler.hpp"
//#include "renderer/spritesRenderer/ParticleSpriteRenderer.cuh"
//#include "renderer/spritesRenderer/InputHandler.hpp"
#include "renderer/spritesRenderer/ParticleSpriteRenderer.cuh"
#include "renderer/spritesRenderer/SpriteRendererInputHandler.hpp"
#include "renderer/sphereRenderer/SphereRenderer.cuh"
#include "Timer.hpp"

#include <thread>

#include "simulations/testSim/RndTestSimCPU.hpp"
#include "simulations/testSim/RndTestSimGPU.cuh"

#include "simulations/gravitySim/GravitySimCPU.hpp"
#include "simulations/gravitySim/GravitySimGPU.cuh"

#define MAX_FRAME_TIME 0.1f

//
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

//    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
//
//    const GLFWvidmode *mode = glfwGetVideoMode(monitor);
//
//    glfwWindowHint(GLFW_RED_BITS, mode->redBits);
//    glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
//    glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
//    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
//    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
//    Renderer_I* renderer = new SphereRenderer(WINDOW_WIDTH, WINDOW_HEIGHT);
    Renderer_I *renderer = new ParticleSpriteRenderer(WINDOW_WIDTH, WINDOW_HEIGHT);

    Camera_I *camera = renderer->getCamera();
    InputHandler_I *inputHandler = renderer->getInputHandler();

    WindowInputHandler windowInputHandler;
    wm->addKeyEventListener(&windowInputHandler);
    wm->addWindowEventListener(camera);
    wm->addKeyEventListener(inputHandler);
    wm->addScrollListener(inputHandler);
    wm->addCursorPositionListener(inputHandler);
    wm->addMouseButtonEventListener(inputHandler);

    //TODO change to constructor?
    renderer->init();
    float planet1fracton = 0.5;
    float planet2fracton = 0.5;
    int num_planet1 = (int) (planet1fracton * NUM_PARTICLES);
    int num_planet2 = (int) (planet2fracton * NUM_PARTICLES);
    assert(num_planet1 + num_planet2 == NUM_PARTICLES);

    Particles *particles = new Particles(NUM_PARTICLES);
    PlanetBuilder::buildPlanet(particles, 0, num_planet1,
                               Particles::TYPE::IRON, (1220.f * 0.25f) / DIST_SCALING,
                               Particles::TYPE::SILICATE, (6371.f * 0.25f) / DIST_SCALING,
            //glm::vec3(0), glm::vec3(0), glm::vec3(0, 7.2921159e-5, 0),
                               glm::vec3(0), glm::vec3(0, 0, 0), glm::vec3(0, 0, 0));
    PlanetBuilder::buildPlanet(particles, num_planet1, num_planet2,
                               Particles::TYPE::IRON, (1220.f * 0.25f) / DIST_SCALING,
                               Particles::TYPE::SILICATE, (6371.f * 0.25f) / DIST_SCALING,
            //glm::vec3(0), glm::vec3(0), glm::vec3(0, 7.2921159e-5, 0),
                               glm::vec3(5000.0f / DIST_SCALING), glm::vec3(0, 0, 0), glm::vec3(0, 0, 0));


//    particles->setParticlePos(renderer->allocateParticlesAndInit_cpu(NUM_PARTICLES, particles->pos));
//    RndTestSimCPU sim(particles);


//    RndTestSimGPU sim(particles, renderer->allocateParticlesAndInit_gpu(NUM_PARTICLES, particles->pos));


    ///CPU GRAVITY
    particles->setParticlePos(renderer->allocateParticlesAndInit_cpu(NUM_PARTICLES, particles->pos));
    GravitySimCPU sim(particles);
    ///\CPU

//    ///GPU GRAVITY
//    GravitySimGPU sim(particles, renderer->allocateParticlesAndInit_gpu(NUM_PARTICLES, particles->pos));
//    ///\GPU

    displayOpenGLInfo();
    Timer timer;
    timer.start();
    for (int i = 0; i < particles->numParticles; ++i) {
        particles->pos[i] += glm::vec4(particles->pos[i].x, particles->pos[i].y,
                                       particles->pos[i].z, 1);
        //TODO
    }

    // Main loop
    while (!wm->shouldClose()) {
        float frameTime = timer.getFrameTime();
        if (frameTime > MAX_FRAME_TIME)
            frameTime = MAX_FRAME_TIME;

        usleep(10000);
        if (windowInputHandler.runSimulation) {
            sim.updateStep(1);
            windowInputHandler.runSimulation = false;
        }


        renderer->render();
        wm->swapBuffers();

        wm->setTitle(windowTitle + " @" + std::to_string(1 / frameTime) + " fps");

    }
    renderer->destroy();
    wm->close();
    return 0;
}
