
#include "common.hpp"


using namespace std;
using namespace glm;

#define MAX_FRAME_TIME 0.1f
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

int main(int argc, char **argv)
{
    GLenum error = glewInit();
    if (error != GLEW_OK) {
        throw std::runtime_error("Can't load GL");
    }
    return 0;
}

