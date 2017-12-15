#ifndef AGP_PROJECT_UTIL_HPP
#define AGP_PROJECT_UTIL_HPP

#include "../common.hpp"

namespace agp
{
    namespace util
    {
        /**
         * Helper method that loads the Vertex and Fragment Shaders, and
         * returns the OpenGL Program associated with them.
         */
        GLuint loadShaders(const char *vertex_shader_filename,
                           const char *fragment_shader_filename);

        /**
         * Helper method that displays information about OpenGL.
         */
        void displayOpenGLInfo();
    }
}

#endif
