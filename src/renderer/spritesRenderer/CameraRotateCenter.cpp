#include "CameraRotateCenter.hpp"

const float PI = 3.14159265358979323846f;


CameraRotateCenter::CameraRotateCenter(int windowWidth, int windowHeight) :
        windowWidth(windowWidth),
        windowHeight(windowHeight) {
    position.x = 0;
    position.y = PI / 4;
    position.z = 50.0;
}

void CameraRotateCenter::applyInput() {
    position.x -= velocity.x;
    position.y -= velocity.y;
    position.z *= (1.0 - velocity.z);
    look_at += look_at_vel;

    velocity *= 0.72; // damping
    look_at_vel *= 0.90;

    // limits
    if (position.x < 0) position.x += 2 * PI;
    if (position.x >= 2 * PI) position.x -= 2 * PI;
    position.y = std::max(-PI / 2 + 0.001f, std::min(position.y, PI / 2 - 0.001f));
}

glm::mat4 CameraRotateCenter::getProj() {
    return glm::infinitePerspective(
            glm::radians(30.0f), windowWidth / (float) windowHeight, 1.f);
}

glm::vec3 getCartesianCoordinates(glm::vec3 v) {
    return glm::vec3(
            cos(v.x) * cos(v.y),
            sin(v.x) * cos(v.y),
            sin(v.y)) * v.z;
}

glm::mat4 CameraRotateCenter::getView() {
    // polar to cartesian coordinates
    glm::vec3 view_pos = getCartesianCoordinates(position);

    return glm::lookAt(
            view_pos + look_at,
            look_at,
            glm::vec3(0, 0, 1));
}

glm::vec3 CameraRotateCenter::getForward() {
    return glm::normalize(-getCartesianCoordinates(position));
}

glm::vec3 CameraRotateCenter::getRight() {
    return glm::normalize(
            glm::cross(
                    getCartesianCoordinates(position),
                    glm::vec3(0, 0, 1)));
}

glm::vec3 CameraRotateCenter::getUp() {
    return glm::normalize(
            glm::cross(
                    getCartesianCoordinates(position),
                    getRight()));
}

void CameraRotateCenter::addVelocity(glm::vec3 vel) {
    velocity += vel;
}

void CameraRotateCenter::addLookAtVelocity(glm::vec3 vel) {
    look_at_vel += vel;
}

glm::vec3 CameraRotateCenter::getPosition() {
    return position;
}

void CameraRotateCenter::onWindowSizeChanged(int width, int height) {
    windowWidth = width;
    windowHeight = height;
}

int CameraRotateCenter::getWindowWidth() {
    return windowWidth;
}

int CameraRotateCenter::getWindowHeight() {
    return windowHeight;
}


