cmake_minimum_required(VERSION 3.5)
if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo -framework GLUT") # compiler version and compiler flags.
endif ()
if (UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif ()


project(makingOfMoon)


set(LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")


set(SOURCE_FILES
        ${SRC_DIR}/util/util.cpp
        ${SRC_DIR}/util/util.hpp
        ${SRC_DIR}/viz/Camera.cpp
        ${SRC_DIR}/viz/Camera.hpp
        ${SRC_DIR}/viz/Sphere.hpp
        ${SRC_DIR}/common.hpp
        ${SRC_DIR}/main.cpp
        ${SRC_DIR}/sim/ParticleSimI.hpp
        ${SRC_DIR}/sim/GravitySim.hpp
        ${SRC_DIR}/sim/GravitySim.cpp
        ${SRC_DIR}/util/Timer.hpp
        ${SRC_DIR}/util/Timer.cpp
        ${SRC_DIR}/sim/Particle.hpp
        ${SRC_DIR}/viz/StaticSphere.cpp
        ${SRC_DIR}/viz/StaticSphere.hpp
        ${SRC_DIR}/viz/ParticleSphere.cpp
        ${SRC_DIR}/viz/ParticleSphere.hpp
        ${SRC_DIR}/viz/Renderer.cpp
        ${SRC_DIR}/viz/Renderer.hpp
        ${SRC_DIR}/viz/RendererInputHandler.hpp
        ${SRC_DIR}/viz/RendererInputHandler.cpp
        ${SRC_DIR}/sim/PlanetBuilder.cpp
        ${SRC_DIR}/sim/PlanetBuilder.hpp
        ${SRC_DIR}/util/WindowManager.hpp
        ${SRC_DIR}/util/WindowManager.cpp
        ${SRC_DIR}/viz/Model.hpp
        ${SRC_DIR}/viz/Model.cpp
        ${SRC_DIR}/viz/GeometryBuilder.hpp
        ${SRC_DIR}/viz/GeometryBuilder.cpp
        )

add_executable(makingOfMoon ${SOURCE_FILES})

# glad
set(GLAD_DIR "${LIB_DIR}/glad")
add_library("glad" "${GLAD_DIR}/src/glad.c")
include_directories(${GLAD_DIR}/include)
target_include_directories("glad" PRIVATE "${GLAD_DIR}/include")
target_include_directories(${PROJECT_NAME} PRIVATE "${GLAD_DIR}/include")
target_link_libraries(${PROJECT_NAME} "glad" "${CMAKE_DL_LIBS}")


# glm
set(GLM_DIR "${LIB_DIR}/glm")
add_subdirectory("${GLM_DIR}")
target_link_libraries(${PROJECT_NAME} "glm" "${GLM_LIBRARIES}")
target_include_directories(${PROJECT_NAME} PRIVATE "${GLM_DIR}/include")


# glfw
if (UNIX AND NOT APPLE)

    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    find_package(glfw3 3.2 REQUIRED)
    target_link_libraries(${PROJECT_NAME} ${CMAKE_DL_LIBS} glfw)
endif ()
if (APPLE)
    set(GLFW_DIR "${LIB_DIR}/glfw3")
    add_subdirectory("${GLFW_DIR}")
    target_link_libraries(${PROJECT_NAME} "glfw" "${GLFW_LIBRARIES}")
    target_include_directories(${PROJECT_NAME} PRIVATE "${GLFW_DIR}/include")
endif ()
