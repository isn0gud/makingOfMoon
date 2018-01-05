#pragma once

#include "common.hpp"
#include "WindowManager.hpp"

class Camera_I : public WindowEventListener {
public:
    virtual void onWindowSizeChanged(int width, int height) = 0;
};
