#ifndef _COMMON_H
#define _COMMON_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "glad/glad.h"
#include <ctime>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext.hpp>
#include <vector>
#include <stdint.h>
#include <glm/vec3.hpp>
#include <chrono>

#define PATH_MAX    4096
#define GL_SUCCESS  0

typedef uint8_t BYTE;

#endif
