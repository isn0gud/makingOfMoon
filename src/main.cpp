#include <iostream>

#include "WindowInputHandler.hpp"
#include "renderer/spritesRenderer/ParticleRenderer.hpp"
#include "renderer/spritesRenderer/InputHandler.hpp"

#include <thread>

#include "simulations/RndTestSim.hpp"

#define MAX_FRAME_TIME 0.1f

using namespace std;

static void errorCallbackFunction(int error, const char *description) {
    cerr << "GLFW-Error: " << description << endl;
}

int main(int argc, char **argv) {

    int WINDOW_WIDTH = 800;
    int WINDOW_HEIGHT = 600;
    int NUM_PARTICLES = 50 * 256;     ///< Number of particles simulated

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

    RndTestSim sim;
    sim.initParticles(particles);


    // Main loop
    while (!wm->shouldClose()) {
        double frame_start = glfwGetTime();

        camera.applyInput();

        sim.updateStep(1);

        renderer.render();

        wm->swapBuffers();
        // Window refresh

        // Thread sleep to match min frame time
        double frame_end = glfwGetTime();
        double elapsed = frame_end - frame_start;
        float frameTime = 1.0f / 60.0f; // 60 fps
        if (elapsed < frameTime) {
            this_thread::sleep_for(chrono::nanoseconds(
                    (long int) ((frameTime - elapsed) * 1000000000)));
        }
    }
    renderer.destroy();
    wm->close();
    return 0;
}