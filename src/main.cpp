
#include "common.hpp"
#include "util/ShaderProgram.hpp"
#include "viz/Camera.hpp"
#include "viz/Sphere.hpp"
#include "sim/ParticleSimI.hpp"
#include "viz/Sphere.hpp"
#include "util/Timer.hpp"

#include "viz/StaticSphere.hpp"
#include "viz/ParticleSphere.hpp"
#include "viz/Sphere.hpp"
#include "viz/ParticleSphereRenderer.hpp"

#include "sim/GravitySim.hpp"
#include "util/WindowManager.hpp"

#include <time.h>


using namespace std;
using namespace glm;
//using namespace agp;

#define MAX_FRAME_TIME 0.1f
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

class InputHandler : public KeyEventListener
{
public:
    bool runSimulation;

    InputHandler() : runSimulation(false) {}

    void onKeyEvent(int key, int scancode, int action, int mods)
    {
        if (action == GLFW_PRESS)
        {
            if (key == GLFW_KEY_ESCAPE)
                WindowManager::getInstance()->singalShouldClose();
            else if(key == GLFW_KEY_SPACE)
                runSimulation = !runSimulation;
        }
    }
};

int main(int argc, char **argv)
{
    // Open window
    WindowManager* wm = WindowManager::getInstance();
    string windowTitle = "AGP Project - The Making of the Moon";
    wm->open(WINDOW_WIDTH, WINDOW_HEIGHT, windowTitle, true);
    srand(time(NULL));

    // Init simulator and create sphere adaptors
    GravitySim sim;
    std::vector<Sphere*> spheres;
    std::vector<Particle*> particles = sim.getParticles();
    for (Particle *p :particles) {
        spheres.push_back(new ParticleSphere(p));
    }

    // Init renderer
    ParticleSphereRenderer renderer;
    renderer.init(WINDOW_WIDTH, WINDOW_HEIGHT);

    InputHandler input;
    input.runSimulation = false;
    wm->addKeyEventListener(&input);

    Timer timer;
    timer.start();

    // Main loop
    while (!wm->shouldClose())
    {
        float frameTime = timer.getFrameTime();
        if (frameTime > MAX_FRAME_TIME)
            frameTime = MAX_FRAME_TIME;

        if (input.runSimulation)
            sim.updateStep(1);
        renderer.render(spheres, frameTime);
        wm->setTitle(windowTitle + " @" + to_string(1/frameTime) + " fps");
    }

    //calls destructor on all elements
//    renderer.clear();
    spheres.clear();

    //sim.clear();

    wm->close();
    return 0;
}

