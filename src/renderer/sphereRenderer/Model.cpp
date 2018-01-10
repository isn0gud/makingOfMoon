#include "Model.hpp"

void Model::loadVertexData(std::vector<glm::vec3> vertices, GLint shader, std::string vertexAttributeName) {
    nVertices = vertices.size();
    vertexBuffer = new float[vertices.size() * 3];
    for (int i = 0; i < nVertices; i++) {
        vertexBuffer[i * 3] = vertices[i].x;
        vertexBuffer[i * 3 + 1] = vertices[i].y;
        vertexBuffer[i * 3 + 2] = vertices[i].z;
    }

    GLint vertexAttributeLocation = glGetAttribLocation(shader, vertexAttributeName.c_str());

    if (glGenVertexArrays == NULL)
    {
        std::cout << "WTF!" << std::endl;
    }

    glGenVertexArrays(1, &vertexArrayObjectId);
    glBindVertexArray(vertexArrayObjectId);
    glGenBuffers(1, &vertexBufferId);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId);
    glBufferData(GL_ARRAY_BUFFER, nVertices * 3 * sizeof(GLfloat), vertexBuffer, GL_STATIC_DRAW);
    glVertexAttribPointer(vertexAttributeLocation, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(vertexAttributeLocation);
}

void Model::drawSolid() {
    glBindVertexArray(vertexArrayObjectId);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId);
    glDrawArrays(GL_TRIANGLES, 0, nVertices);
}

void Model::drawWireframe() {
    glBindVertexArray(vertexArrayObjectId);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawArrays(GL_TRIANGLES, 0, nVertices);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void Model::drawSolidInstanced(int numberOfInstances) {
    glBindVertexArray(vertexArrayObjectId);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId);
    glDrawArraysInstanced(GL_TRIANGLES, 0, nVertices, numberOfInstances);
}

void Model::drawWireframeInstanced(int numberOfInstances) {
    glBindVertexArray(vertexArrayObjectId);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawArraysInstanced(GL_TRIANGLES, 0, nVertices, numberOfInstances);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void Model::freeVertexData() {
    glDeleteBuffers(1, &vertexBufferId);
    delete[] vertexBuffer;
}
