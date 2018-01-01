#pragma once

#include "common.hpp"
#include "WindowManager.hpp"

class Camera : public WindowEventListener {
public:
    virtual int getWindowWidth() = 0;

    virtual int getWindowHeight() = 0;

    virtual void onWindowSizeChanged(int width, int height) = 0;

    virtual void applyInput()=0;

    virtual glm::mat4 getProj() = 0;

    virtual glm::mat4 getView() = 0;


};
