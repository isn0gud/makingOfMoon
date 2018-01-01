#include <iostream>

#include "WindowInputHandler.hpp"
#include "renderer/spritesRenderer/ParticleRenderer.hpp"
#include "renderer/spritesRenderer/InputHandler.hpp"
#include "Timer.hpp"

#include <thread>

//#include "simulations/testSim/RndTestSim.hpp"
#include "simulations/gravitySim/GravitySim.hpp"

#define MAX_FRAME_TIME 0.1f

using namespace std;

int main(int argc, char **argv) {

    int WINDOW_WIDTH = 800;
    int WINDOW_HEIGHT = 600;
//    int NUM_PARTICLES = 50 * 256;     ///< Number of particles simulated
    int NUM_PARTICLES = 500;

    // Open window
    WindowManager *wm = WindowManager::getInstance();
    string windowTitle = "AGP Project - The Making of the Moon";
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
    CameraRotateCenter camera(WINDOW_WIDTH, WINDOW_HEIGHT);
    InputHandler inputHandler(&camera);
    ParticleRenderer renderer(&camera);


    WindowInputHandler windowInputHandler;
    wm->addKeyEventListener(&windowInputHandler);
    wm->addWindowEventListener(&camera);

    wm->addKeyEventListener(&inputHandler);
    wm->addScrollListener(&inputHandler);
    wm->addCursorPositionListener(&inputHandler);
    wm->addMouseButtonEventListener(&inputHandler);

    //TODO change to constructor?
    renderer.init();

    Particles *particles = renderer.allocateParticles(NUM_PARTICLES);

    GravitySim sim;
    sim.initParticles(particles);


    Timer timer;
    timer.start();

    // Main loop
    while (!wm->shouldClose()) {
        float frameTime = timer.getFrameTime();
        if (frameTime > MAX_FRAME_TIME)
            frameTime = MAX_FRAME_TIME;

        camera.applyInput();

        sim.updateStep(1);

        renderer.render();

        wm->swapBuffers();
        // Window refresh

        wm->setTitle(windowTitle + " @" + to_string(1 / frameTime) + " fps");

    }
    renderer.destroy();
    wm->close();
    return 0;
}