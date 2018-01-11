#pragma once


#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <ctime>

#define GLEW_STATIC

#include <GL/glew.h>


#define GLFW_INCLUDE_NONE

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL ///not compatible with cuda

#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
//#include <glm/ext.hpp> ///not compatible with cuda
#include <vector>
#include <stdint.h>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <chrono>

#define PATH_MAX  4096
#define GL_SUCCESS 0

#define G_ 2.0f          ///< Gravitational parameter
#define FRAME_TIME 1.0f / 60.0f // 60 fps       ///< Minimum frame time in seconds
#define SIM_dt 0.005          ///< Simulation delta t
#define MAX_ITER_PER_FRAME 4  ///< Simulation iterations per frame rendered
#define DAMPING 0.999998        ///< Damping parameter for simulating 'soupy' galaxy (1.0 no damping)
#define NUM_PARTICLES 1000
#define BLUR true

#define MASS_SCALING 1.0f
#define DIST_SCALING 1.0f

typedef uint8_t BYTE;


