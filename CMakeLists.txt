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


set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/nbody")


set(SOURCE_FILES
        src/render/shaderUtil.cpp
        src/render/shaderUtil.hpp
        src/render/ParticleRenderer.cpp
        src/render/ParticleRenderer.hpp
        src/render/Camera.cpp
        src/render/Camera.hpp
        src/sim/gen.cpp
        src/sim/gen.hpp
        src/main.cpp
        )



add_executable(makingOfMoon ${SOURCE_FILES})

target_compile_features(makingOfMoon PRIVATE cxx_auto_type cxx_nullptr cxx_range_for)

target_include_directories(makingOfMoon PRIVATE
        ${GLM_INCLUDE_DIRS}
        ${GLFW_INCLUDE_DIR}
        ${GLEW_INCLUDE_DIR}
        ${OPENGL_INCLUDE_DIR})

target_link_libraries(makingOfMoon ${GLFW_LIBRARIES} ${GLEW_LIBRARY} ${OPENGL_gl_LIBRARY})


# glad
#set(GLAD_DIR "${LIB_DIR}/glad")
#add_library("glad" "${GLAD_DIR}/src/glad.c")
#include_directories(${GLAD_DIR}/include)
#target_include_directories("glad" PRIVATE "${GLAD_DIR}/include")
#target_include_directories(${PROJECT_NAME} PRIVATE "${GLAD_DIR}/include")
#target_link_libraries(${PROJECT_NAME} "glad" "${CMAKE_DL_LIBS}")

#
## glm
#set(GLM_DIR "${LIB_DIR}/glm")
#add_subdirectory("${GLM_DIR}")
#target_link_libraries(${PROJECT_NAME} "glm" "${GLM_LIBRARIES}")
#target_include_directories(${PROJECT_NAME} PRIVATE "${GLM_DIR}/include")
#
#
## glfw
#set(GLFW_DIR "${LIB_DIR}/glfw3")
#add_subdirectory("${GLFW_DIR}")
#target_link_libraries(${PROJECT_NAME} "glfw" "${GLFW_LIBRARIES}")
#target_include_directories(${PROJECT_NAME} PRIVATE "${GLFW_DIR}/include")
#
#
#
## glew
#
##set(GLEW_DIR "${LIB_DIR}/glew-2.1.0/build/cmake")
##add_subdirectory("${GLEW_DIR}")
##target_link_libraries(${PROJECT_NAME} "glew" "${GLEW_LIBRARIES}")
##target_include_directories(${PROJECT_NAME} PRIVATE "${GLEW_DIR}/include")
#
#
#
#list(APPEND CMAKE_PREFIX_PATH "lib/glew-2.1.0")
#find_package(GLEW REQUIRED)
#if (GLEW_FOUND)
#    include_directories(${GLEW_INCLUDE_DIRS})
#    link_libraries(${GLEW_LIBRARIES})
#endif()
