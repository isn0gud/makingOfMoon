//
// Created by Pius Friesch on 18/11/2017.
//

#ifndef ASS_OPENGL_CAMERA_H
#define ASS_OPENGL_CAMERA_H

#include "../common.hpp"

class Camera {

public:

#define WINDOW_WIDTH 800.0f
#define WINDOW_HEIGHT 600.0f
#define ROTATION_SPEED 0.002f
#define MOVEMENT_SPEED 1.f
#define INIT_CAMERA_DIST 2.5f

private:
    float rot_speed = ROTATION_SPEED;
    float m_speed = MOVEMENT_SPEED;

    static glm::mat4 getPerspective(float width, float height) {
        return glm::perspective(45.0f, width / height, 0.1f, 1000.0f);
    }

    glm::vec3 m_up = glm::vec3(0.0f, 1.0f, 0.0f); // y is up
    glm::mat4 proj = getPerspective(WINDOW_WIDTH, WINDOW_HEIGHT);
    glm::vec3 m_position;
    glm::vec3 m_direction;


    glm::mat4 getViewMat();


    void translate(glm::vec3 &direction);

    void rotate(float amount, glm::vec3 axis);

public:

    void updateWinShape(float width, float height);

    glm::vec2 mouse_pos = glm::vec2(0.0f);

    //init the camera looking to the center from INIT_CAMERA_DIST away in x direction
    Camera() {
        m_position = glm::vec3(0, 0.0f, INIT_CAMERA_DIST);
        m_direction = glm::vec3(0.0f, 0.0f, -1.0f);
    }


    glm::mat4 getViewMVP(glm::mat4 model);

    enum MovementType {
        FORWARD, BACKWARD, STRAFE_LEFT, STRAFE_RIGHT
    };


    void applyMovement(MovementType movement);

    void applyMovementMouse(double x, double y);

};


#endif //ASS_OPENGL_CAMERA_H
