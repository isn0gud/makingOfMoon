#pragma once

#include "../../common.hpp"
#include <vector>


class Model {
    int nVertices;
    float *vertexBuffer;
    GLuint vertexBufferId;
    GLuint vertexArrayObjectId;

public:

    void loadVertexData(std::vector<glm::vec3> vertices, GLint shader, std::string vertexAttributeName);

    void drawSolid();
    void drawWireframe();

    void drawSolidInstanced(int numberOfInstances);
    void drawWireframeInstanced(int numberOfInstances);

    void freeVertexData();
};

