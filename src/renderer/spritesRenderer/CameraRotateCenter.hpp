#pragma once

#include "../../common.hpp"
#include "../../Camera_I.hpp"

class CameraRotateCenter : public Camera_I {

public:
    int getWindowWidth() override;

    int getWindowHeight() override;

    CameraRotateCenter(int windowWidth, int windowHeight);


    /**
     * Computes projection matrix from camera parameters
     * @param c camera parameters
     * @param width viewport width
     * @param height viewport height
     * @return projection matrix
     */
    glm::mat4 getProj() override;

    /**
     * Computes view matrix from camera parameters
     * @param c camera parameters
     * @param view matrix
     */
    glm::mat4 getView() override;

    glm::vec3 getForward();

    glm::vec3 getRight();

    glm::vec3 getUp();

    glm::vec3 getPosition();

    void addVelocity(glm::vec3 vel);

    void addLookAtVelocity(glm::vec3 vel);

    void onWindowSizeChanged(int width, int height) override;

    /**
     * Computes next step of camera parameters
     * @param c camera at step n
     * @return camera at step n+1
     */
    void applyInput() override;


private:
    int windowWidth;
    int windowHeight;
    glm::vec3 position;    ///< Polar coordinates in radians
    glm::vec3 velocity;    ///< dp/dt of polar coordinates
    glm::vec3 look_at;     ///< Where is the camera looking at
    glm::vec3 look_at_vel; ///< dp/dt of lookat position




};

