//
// Created by Pius Friesch on 18/11/2017.
//

#include "Camera.hpp"

Camera::Camera()
{
    position = glm::vec3(0.0f, 0.0f, 0.0f);
    setOrientation(glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    fieldOfView = 45.f;
    aspectRatio = 800/600;
    near = 1;
    far = 100000;

    projectionMatrix = glm::perspective(fieldOfView, aspectRatio, near, far);
}

void Camera:: updateWindowShape(float width, float height)
{
    aspectRatio = width / height;
    projectionMatrix = glm::perspective(fieldOfView, aspectRatio, near, far);
}

void Camera::setFrustrum(float fov, float near, float far)
{
    fieldOfView = fov;
    this->near = near;
    this->far = far;
    projectionMatrix = glm::perspective(fieldOfView, aspectRatio, near, far);
}

void Camera::setProjectionMatrix(float fov, float width, float height, float near, float far)
{
    fieldOfView = fov;
    aspectRatio = width / height;
    this->near = near;
    this->far = far;
    projectionMatrix = glm::perspective(fieldOfView, aspectRatio, near, far);
}

glm::mat4 Camera::getProjectionMatrix()
{
    return projectionMatrix;
}

void Camera::setOrientation(glm::vec3 forward, glm::vec3 up)
{
    forward = glm::normalize(forward);
    glm::vec3 right = cross(forward, up);
    up = glm::normalize(cross(right, forward));
    right = glm::normalize(right);

    orientation[0] = right;
    orientation[1] = up;
    orientation[2] = forward * (-1);
}

glm::mat4 Camera::getViewTransformationMatrix()
{
    glm::mat4 result;
    result[0] = glm::vec4(orientation[0][0], orientation[0][1], orientation[0][2], 0);
    result[1] = glm::vec4(orientation[1][0], orientation[1][1], orientation[1][2], 0);
    result[2] = glm::vec4(orientation[2][0], orientation[2][1], orientation[2][2], 0);
    result[3] = glm::vec4(position.x,        position.y,        position.z,        1);
    return glm::affineInverse(result);
}

glm::mat4 Camera::getModelViewProjectionMatrix(glm::mat4 modelTransformationMatrix)
{
    return projectionMatrix * getViewTransformationMatrix() * modelTransformationMatrix;
}
