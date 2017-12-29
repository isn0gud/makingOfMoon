
#include "common.hpp"
#include "WindowManager.hpp"

using namespace std;
using namespace glm;

#define MAX_FRAME_TIME 0.1f
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

int main(int argc, char **argv)
{

    // Open window
    WindowManager* wm = WindowManager::getInstance();
    string windowTitle = "AGP Project - The Making of the Moon";
    wm->open(WINDOW_WIDTH, WINDOW_HEIGHT, windowTitle, true);

    return 0;
}

