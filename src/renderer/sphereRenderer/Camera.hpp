#pragma once

#include "../../common.hpp"

class Camera {
private:
    float fieldOfView;
    float aspectRatio;
    float near;
    float far;

    glm::mat4 projectionMatrix;
public:
    Camera();

    glm::vec3 position;
    glm::mat3 orientation;

    void setOrientation(glm::vec3 forward, glm::vec3 up);

    glm::mat4 getViewTransformationMatrix();

    void updateWindowShape(float width, float height);

    void setFrustrum(float fov, float near, float far);

    void setProjectionMatrix(float fov, float windowWidth, float windowHeight, float near, float far);

    glm::mat4 getProjectionMatrix();

    glm::mat4 getModelViewProjectionMatrix(glm::mat4 modelTransformationMatrix);
};

