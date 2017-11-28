
#include "common.hpp"
#include "util/sphereGLUT.hpp"
#include "util/util.hpp"
#include "util/Camera.hpp"
#include "util/Sphere.hpp"
#include "sim/ParticleSimI.hpp"
#include "sim/StaticVecFieldRndSim.hpp"
#include "util/Sphere.hpp"
#include "util/Timer.hpp"

#include "util/StaticSphere.hpp"
#include "util/ParticleSphere.hpp"
#include "sim/RndAccelFieldSim.hpp"
#include "sim/GravitySim.hpp"
using namespace std;
using namespace glm;
using namespace agp;
//using namespace agp::glut;

GLuint g_default_vao = 0;

void init() {
    // Generate and bind the default VAO
    glGenVertexArrays(1, &g_default_vao);
    glBindVertexArray(g_default_vao);

    // Set the background color (RGBA)
    glClearColor(0.3f, 0.3f, 0.3f, 0.0f);

    // Your OpenGL settings, such as alpha, depth and others, should be
    // defined here! For the assignment, we only ask you to enable the
    // alpha channel.

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void release() {
    // Release the default VAO
    glDeleteVertexArrays(1, &g_default_vao);

    // Do not forget to release any memory allocation here!
}


void dispSphere(Camera cam, Sphere *s, GLuint shaderProgram, Sphere::Shape shape) {

//    switch (shape) {
//        case Sphere::Shape::WIRE:
//            s->setAlpha(0.2f);
//            break;
//        case Sphere::Shape::SOLID:
//            s->setAlpha(0.2f);
//            break;
//    }
    GLint uniColor = glGetUniformLocation(shaderProgram, "inColor");
    glUniform4fv(uniColor, 1, glm::value_ptr(s->getColor()));
    GLint uniMvp = glGetUniformLocation(shaderProgram, "mvp");
    glm::mat4 mvp = cam.getViewMVP(s->getModel());
    glUniformMatrix4fv(uniMvp, 1, GL_FALSE, glm::value_ptr(mvp));
    switch (shape) {
        case Sphere::Shape::WIRE:
            agp::glut::glutWireSphere(s->getRadius(), s->getSlices(), s->getStacks());
            break;
        case Sphere::Shape::SOLID:
            agp::glut::glutSolidSphere(s->getRadius(), s->getSlices(), s->getStacks());
            break;
    }

}


void display(GLFWwindow *window, Camera cam, std::vector<Sphere *> spheres, GLuint shaderProgram) {
    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//    printf("GLFW triggered the display() callback!\n");


    for (Sphere *s : spheres) {
        dispSphere(cam, s, shaderProgram, Sphere::Shape::SOLID);
        dispSphere(cam, s, shaderProgram, Sphere::Shape::WIRE);
    }

    // Swap buffers and force a redisplay
    glfwSwapBuffers(window);
    glfwPollEvents();
}


Camera *cam = nullptr;
ParticleSimI *sim = nullptr;
bool stopSim = true;


void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, true);
        } else if (key == GLFW_KEY_UP) {
            if (cam)
                cam->applyMovement(Camera::FORWARD);
        } else if (key == GLFW_KEY_DOWN) {
            if (cam)
                cam->applyMovement(Camera::BACKWARD);
        } else if (key == GLFW_KEY_LEFT) {
            if (cam)
                cam->applyMovement(Camera::STRAFE_RIGHT);
        } else if (key == GLFW_KEY_RIGHT) {
            if (cam)
                cam->applyMovement(Camera::STRAFE_LEFT);
        } else if (key == GLFW_KEY_SPACE) {
            stopSim = !stopSim;
        }
    }
}

bool mousePressed = false;

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
    if (action == GLFW_PRESS) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, 1);

        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        cam->mouse_pos = glm::vec2(xpos, ypos);
        mousePressed = true;
    }
//    else if (action == GLFW_RELEASE) {
//        mousePressed = false;
//    }
}

static void cursor_position_callback(GLFWwindow *window, double xpos, double ypos) {
    if (mousePressed) {
        glfwGetCursorPos(window, &xpos, &ypos);
        cam->applyMovementMouse(xpos, ypos);
    }
}

void window_size_callback(GLFWwindow *window, int width, int height) {
    if (cam) {
        cam->updateWinShape(width, height);
    }

}

int main(int argc, char **argv) {
    GLFWwindow *window = NULL;

    // Initialize GLFW
    if (!glfwInit()) {
        return GL_INVALID_OPERATION;
    }

    // Setup the OpenGL context version
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open the window and create the context
    window = glfwCreateWindow(800, 600, "Applied GPU Programming", NULL, NULL);

    if (window == NULL) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetWindowSizeCallback(window, window_size_callback);
    // Capture the input events (e.g., keyboard)
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Init GLAD to be able to access the OpenGL API
    if (!gladLoadGL()) {
        return GL_INVALID_OPERATION;
    }

    // Display OpenGL information
    util::displayOpenGLInfo();

    // Initialize the 3D view
    init();

    GLuint shaderProgram = util::loadShaders("shaders/vertexShader.glsl", "shaders/fragmentShader.glsl");

    glUseProgram(shaderProgram);

    sim = new GravitySim();

    std::vector<Sphere *> spheres;
// Sun
//    spheres.push_back(new StaticSphere(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec4(1.0f, 0.0f, 0.0f, 0.2f), 0.1f));

    cam = new Camera();

    Timer timer;
    timer.start();

    std::vector<Particle *> particles = sim->getParticles();
    for (Particle *p :particles) {
        spheres.push_back(new ParticleSphere(p));
    }

    int i = 0;
    // Launch the main loop for rendering
    while (!glfwWindowShouldClose(window)) {
        if (!stopSim && sim) {
            float frameTime = timer.getFrameTime();
            if(frameTime > 0.1)
                frameTime = 0.1;
            sim->updateStep(1, frameTime);
//            stopSim = !stopSim;
        }
        display(window, *cam, spheres, shaderProgram);

        i++;
    }
    //calls destructor on all elements
    spheres.clear();
    // Release all the allocated memory
    release();

    // Release GLFW
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

