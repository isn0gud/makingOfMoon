#pragma once

#include "../../common.hpp"
#include <vector>




class Model {
    int nVertices;
    float *vertexBuffer;
    GLuint vertexBufferId;
    GLuint vertexArrayObjectId;

public:

    void loadVertexData(vector<vec3> vertices, GLint shader, string vertexAttributeName);

    void drawSolid();

    void drawWireframe();

    void freeVertexData();
};

