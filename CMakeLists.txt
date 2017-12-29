cmake_minimum_required(VERSION 3.5)


enable_language(CXX)

#set(CMAKE_INSTALL_PREFIX "lib/builds")

#set(GLEW_LOCATION "lib/glew-2.1.0/glew_compiled/usr") # compile with "make install DESTDIR=glew_compiled"

#set(OPENGL_INCLUDE_DIR "lib/glfw/include")

#set(GLFW_LOCATION "lib/glfw/glfw_compiled/usr/local") # compile with "make install DESTDIR=glfw_compiled"

#set(GLM_INCLUDE_DIR "lib/glm")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(GLFW REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLM REQUIRED)
find_package(OpenGL REQUIRED)


project(makingOfMoon)


set(SOURCE_FILES
        src/common.hpp
        src/main.cpp
        src/WindowManager.hpp
        src/WindowManager.cpp

        )

add_executable(makingOfMoon ${SOURCE_FILES})

target_compile_features(makingOfMoon PRIVATE cxx_auto_type cxx_nullptr cxx_range_for)

target_include_directories(makingOfMoon PRIVATE
        ${GLM_INCLUDE_DIRS}
        ${GLFW_INCLUDE_DIR}
        ${GLEW_INCLUDE_DIR}
        ${OPENGL_INCLUDE_DIR})

target_link_libraries(makingOfMoon ${GLFW_LIBRARIES} ${GLEW_LIBRARY} ${OPENGL_gl_LIBRARY})

