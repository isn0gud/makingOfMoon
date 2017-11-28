//
// Created by Pius Friesch on 18/11/2017.
//

#include "Camera.hpp"


glm::mat4 Camera::getViewMat() {
    return glm::lookAt(m_position, m_position + m_direction, m_up);
}

void Camera::rotate(float amount, glm::vec3 axis) {
    m_direction = glm::rotate(m_direction, amount, axis);

}

void Camera::translate(glm::vec3 &direction) {
    m_position += direction;

}

glm::mat4 Camera::getViewMVP(glm::mat4 model) {
    return proj * this->getViewMat() * model;
}

void Camera::applyMovement(Camera::MovementType movement) {
    switch (movement) {
        case FORWARD:
            m_position += m_direction * m_speed;
            break;
        case BACKWARD:
            m_position -= m_direction * m_speed;
            break;
        case STRAFE_LEFT:
            m_position += glm::cross(m_direction * m_speed, m_up);
            break;
        case STRAFE_RIGHT:
            m_position -= glm::cross(m_direction * m_speed, m_up);
            break;
    }
}

void Camera::applyMovementMouse(double xpos, double ypos) {
    glm::vec2 delta = mouse_pos - glm::vec2(xpos, ypos);
    rotate(delta.x * rot_speed, m_up);
    rotate(delta.y * rot_speed, glm::cross(m_direction, m_up));

    mouse_pos = glm::vec2(xpos, ypos);
}


void Camera::updateWinShape(float width, float height) {
    proj = getPerspective(width, height);
}


