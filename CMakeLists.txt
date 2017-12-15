cmake_minimum_required(VERSION 3.5)


enable_language(CXX)

set(CMAKE_INSTALL_PREFIX "lib/builds")

set(GLEW_LOCATION "lib/glew-2.1.0")

set(OPENGL_INCLUDE_DIR "lib/glfw-3.2.1/include")

set(GLFW_LOCATION "lib/glfw-3.2.1/glfw_compiled/usr/local") # compile with "make install DESTDIR=glfw_compiled"

set(GLM_INCLUDE_DIR "lib/glm")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(GLM REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)


project(makingOfMoon)


set(SOURCE_FILES
        src/util/util.cpp
        src/util/util.hpp
        src/viz/Camera.cpp
        src/viz/Camera.hpp
        src/viz/Sphere.hpp
        src/common.hpp
        src/main.cpp
        src/sim/ParticleSimI.hpp
        src/sim/GravitySim.hpp
        src/sim/GravitySim.cpp
        src/util/Timer.hpp
        src/util/Timer.cpp
        src/sim/Particle.hpp
        src/viz/StaticSphere.cpp
        src/viz/StaticSphere.hpp
        src/viz/ParticleSphere.cpp
        src/viz/ParticleSphere.hpp
        src/viz/Renderer.cpp
        src/viz/Renderer.hpp
        src/viz/RendererInputHandler.hpp
        src/viz/RendererInputHandler.cpp
        src/sim/PlanetBuilder.cpp
        src/sim/PlanetBuilder.hpp
        src/util/WindowManager.hpp
        src/util/WindowManager.cpp
        src/viz/Model.hpp
        src/viz/Model.cpp
        src/viz/GeometryBuilder.hpp
        src/viz/GeometryBuilder.cpp
        )

add_executable(makingOfMoon ${SOURCE_FILES})

target_compile_features(makingOfMoon PRIVATE cxx_auto_type cxx_nullptr cxx_range_for)

target_include_directories(makingOfMoon PRIVATE
        ${GLM_INCLUDE_DIRS}
        ${GLFW_INCLUDE_DIR}
        ${GLEW_INCLUDE_DIR}
        ${OPENGL_INCLUDE_DIR})

target_link_libraries(makingOfMoon ${GLFW_LIBRARIES} ${GLEW_LIBRARY} ${OPENGL_gl_LIBRARY})

