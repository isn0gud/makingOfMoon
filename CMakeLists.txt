cmake_minimum_required(VERSION 3.9)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(makingOfMoon LANGUAGES CXX CUDA)

set(CMAKE_INSTALL_PREFIX "lib/builds")
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")


set(GLEW_LOCATION "lib/glew-2.1.0/glew_compiled/usr") # compile with "make install DESTDIR=glew_compiled"

set(OPENGL_INCLUDE_DIR "lib/glfw/include")

set(GLFW_LOCATION "lib/glfw/glfw_compiled/usr/local") # compile with "make install DESTDIR=glfw_compiled"

set(GLM_INCLUDE_DIR "lib/glm")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Pass options to NVCC
set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3 -std=c++11
        #        -gencode arch=compute_30,code=sm_30
        #        -gencode arch=compute_35,code=sm_35
        #        -gencode arch=compute_37,code=sm_37
        #        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_52,code=sm_52
        #        -gencode arch=compute_60,code=sm_60
        #        -gencode arch=compute_61,code=sm_61
        #        -gencode arch=compute_70,code=sm_70
        #        -gencode arch=compute_70,code=compute_70
)


set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CUDA_HOST_COMPILER usr/bin/g++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    find_package(glfw3 3.2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLM REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)


set(SOURCE_FILES
        src/simulations/testSim/RndTestSimCPU.hpp
        src/simulations/testSim/RndTestSimCPU.cpp
        src/simulations/gravitySim/GravitySimCPU.hpp
        src/simulations/gravitySim/GravitySimCPU.cpp
        src/simulations/gravitySim/GravitySimGPU.cuh
        src/simulations/gravitySim/GravitySimGPU.cu
        src/simulations/PlanetBuilder.hpp
        src/simulations/PlanetBuilder.cpp

        src/renderer/spritesRenderer/CameraRotateCenter.hpp
        src/renderer/spritesRenderer/CameraRotateCenter.cpp
        src/renderer/spritesRenderer/ParticleSpriteRenderer.cuh
        src/renderer/spritesRenderer/ParticleSpriteRenderer.cu


        src/renderer/sphereRenderer/SphereRenderer.cuh
        src/renderer/sphereRenderer/SphereRenderer.cu
        src/renderer/sphereRenderer/Model.hpp
        src/renderer/sphereRenderer/Model.cpp
        src/renderer/sphereRenderer/Camera.hpp
        src/renderer/sphereRenderer/Camera.cpp
        src/renderer/sphereRenderer/GeometryBuilder.hpp
        src/renderer/sphereRenderer/GeometryBuilder.cpp
        src/renderer/sphereRenderer/SphereRendererInputHandler.hpp
        src/renderer/sphereRenderer/SphereRendererInputHandler.cpp

        src/util/helper_cuda.h
        src/util/helper_string.h

        src/common.hpp
        src/Timer.cpp
        src/Timer.hpp
        src/ShaderProgram.hpp
        src/ShaderProgram.cpp
        src/Particles.hpp
        src/WindowManager.hpp
        src/WindowManager.cpp
        src/renderer/spritesRenderer/SpriteRendererInputHandler.hpp
        src/renderer/spritesRenderer/SpriteRendererInputHandler.cpp
        src/ParticleSim_I.hpp
        src/Renderer_I.hpp
        src/Camera_I.hpp
        src/InputHandler_I.hpp
        src/WindowInputHandler.hpp
        src/main.cpp
        src/simulations/testSim/RndTestSimGPU.cu
        src/simulations/testSim/RndTestSimGPU.cuh
        )

cuda_add_executable(makingOfMoon ${SOURCE_FILES})
#add_executable(makingOfMoon ${SOURCE_FILES})

target_compile_features(makingOfMoon PRIVATE cxx_std_11 cxx_auto_type cxx_nullptr cxx_range_for)

target_include_directories(makingOfMoon PRIVATE
        ${GLM_INCLUDE_DIRS}
        ${GLFW_INCLUDE_DIR}
        ${GLEW_INCLUDE_DIR}
        ${OPENGL_INCLUDE_DIR})

target_link_libraries(makingOfMoon ${CMAKE_DL_LIBS} glfw ${GLFW_LIBRARIES} ${GLEW_LIBRARY} ${OPENGL_gl_LIBRARY} )

