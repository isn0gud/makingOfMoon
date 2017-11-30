#include <iostream>

#include "render/ParticleRenderer.hpp"

#include <glm/glm.hpp>
#include <vector>
#include <cstdlib>

#include <chrono>
#include <thread>

#include "sim/gen.hpp"
#include "render/Camera.hpp"

double scroll = 0;

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    scroll += yoffset;
}

using namespace std;


float mouse_sensibility = 0.002;
float mouse_move_speed = 0.00005;


int main(int argc, char **argv) {


//    SimParam params;
//    params.parseArgs(argc, argv);

    // Window initialization
    GLFWwindow *window;

    if (!glfwInit()) return -1;

    GLFWmonitor *monitor = glfwGetPrimaryMonitor();

    const GLFWvidmode *mode = glfwGetVideoMode(monitor);

    glfwWindowHint(GLFW_RED_BITS, mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    ParticleRenderer renderer;

    renderer.initWindow();

    int width = mode->width;
    int height = mode->height - 30;
    width = 800;
    height = 600;
    window = glfwCreateWindow(width, height, "N-Body Simulation", NULL, NULL);

    glfwMakeContextCurrent(window);
    glfwSetScrollCallback(window, scroll_callback);

    renderer.init(window, width, height);

    // Pos&vel init
    {
        vector<glm::vec4> particlesPos(NUM_PARTICLES);
        vector<glm::vec4> particlesVel(NUM_PARTICLES);
        for (size_t i = 0; i < NUM_PARTICLES; ++i) {
            glm::vec4 p = randomParticlePos();
            glm::vec4 v = randomParticleVel(p);

            particlesPos[i] = p;
            particlesVel[i] = v;
        }
        renderer.populateParticles(particlesPos, particlesVel);
    }
    //TODO setup particles


    Camera camera;

    // Mouse information
    double last_xpos, last_ypos;
    bool drag = false;
    glfwGetCursorPos(window, &last_xpos, &last_ypos);

    // Main loop
    while (!glfwWindowShouldClose(window) &&
           glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_RELEASE) {
        double frame_start = glfwGetTime();

        // View movement
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)) {
            if (!drag) {
                glfwGetCursorPos(window, &last_xpos, &last_ypos);
                drag = true;
            }
            double xpos, ypos;
            double xdiff, ydiff;
            glfwGetCursorPos(window, &xpos, &ypos);
            xdiff = xpos - last_xpos;
            ydiff = ypos - last_ypos;

            last_xpos = xpos;
            last_ypos = ypos;

            camera.addVelocity(glm::vec3(xdiff, -ydiff, 0) * mouse_sensibility);
        } else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)) {
            if (!drag) {
                glfwGetCursorPos(window, &last_xpos, &last_ypos);
                drag = true;
            }
            double xpos, ypos;
            double xdiff, ydiff;
            glfwGetCursorPos(window, &xpos, &ypos);
            xdiff = xpos - last_xpos;
            ydiff = ypos - last_ypos;

            last_xpos = xpos;
            last_ypos = ypos;

            glm::vec3 xvec = camera.getRight();
            xvec.z = 0.0;
            xvec = glm::normalize(xvec);

            glm::vec3 yvec = camera.getUp();
            yvec.z = 0.0;
            yvec = glm::normalize(yvec);

            float dist = camera.getPosition().z;

            camera.addLookAtVelocity(
                    -(float) xdiff * mouse_move_speed * dist * xvec
                    + (float) ydiff * mouse_move_speed * dist * yvec);
        } else drag = false;

        // Scrolling
        camera.addVelocity(glm::vec3(0, 0, scroll * 0.02));
        scroll = 0;

        camera.step();

//        renderer.stepSim();
        //TODO STEP
        renderer.render(camera.getProj(width, height), camera.getView());

        // Window refresh
        glfwSwapBuffers(window);
        glfwPollEvents();

        // Thread sleep to match min frame time
        double frame_end = glfwGetTime();
        double elapsed = frame_end - frame_start;
        float frameTime = 1.0f / 60.0f; // 60 fps
        if (elapsed < frameTime) {
            //TODO thread sleep???
            this_thread::sleep_for(chrono::nanoseconds(
                    (long int) ((frameTime - elapsed) * 1000000000)));
        }
    }
    renderer.destroy();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}