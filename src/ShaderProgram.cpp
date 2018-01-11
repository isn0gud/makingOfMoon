#include "ShaderProgram.hpp"

#include <sstream>
#include <fstream>

ShaderProgram::ShaderProgram() : id(0) {}

void ShaderProgram::source(GLenum shader_type, const std::string &filename) { // loadShaders
    if (!id) id = glCreateProgram();

    std::string code;

    // IO stuff
    try {
        std::stringstream sstream;
        {
            std::ifstream stream;
            stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            stream.open(filename);
            sstream << stream.rdbuf();
        }
        code = sstream.str();
    }
    catch (std::ifstream::failure &e) {
        throw std::runtime_error(std::string("Can't open ") + filename + std::string(e.what()));
    }

    GLint success;
    GLchar info_log[2048];
    const char *shader_data = code.c_str();

    // OpenGL stuff
    GLuint shad_id = glCreateShader(shader_type);
    glShaderSource(shad_id, 1, &shader_data, nullptr);
    glCompileShader(shad_id);
    glGetShaderiv(shad_id, GL_COMPILE_STATUS, &success);
    if (!success) {
        // error log
        glGetShaderInfoLog(shad_id, sizeof(info_log), nullptr, info_log);
        throw std::runtime_error(std::string("Can't compile ") + filename + " " + info_log);
    }
    glAttachShader(id, shad_id);
}

void ShaderProgram::link() {
    GLint success;
    GLchar info_log[2048];

    glLinkProgram(id);
    glGetProgramiv(id, GL_LINK_STATUS, &success);
    if (!success) {
        // error log
        glGetProgramInfoLog(id, sizeof(info_log), nullptr, info_log);
        throw std::runtime_error(std::string("Can't link ") + std::string(info_log));
    }
}

GLuint ShaderProgram::getId() { return id; }
