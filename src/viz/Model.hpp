#ifndef AGP_PROJECT_MESH_HPP
#define AGP_PROJECT_MESH_HPP

#include "../common.hpp"
#include <vector>

using namespace std;
using namespace glm;

class Model
{
    int nVertices;
    float* vertexBuffer;
    GLuint vertexBufferId;
    GLuint vertexArrayObjectId;

public:

    void loadVertexData(vector<vec3> vertices, GLint shader, string vertexAttributeName);
    void drawSolid();
    void drawWireframe();
    void freeVertexData();
};

#endif
