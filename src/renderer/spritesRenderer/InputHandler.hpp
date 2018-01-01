#pragma once

#include "../../common.hpp"
#include "../../WindowManager.hpp"
#include "CameraRotateCenter.hpp"


// class that handles the user input of the renderer and makes it available in a more easily processed format
class InputHandler
        : public KeyEventListener,
          public MouseButtonEventListener,
          public CursorPositionListener,
          public ScrollListener {
private:
    bool mouseIsPressed;
    bool wKeyIsPressed;
    bool aKeyIsPressed;
    bool sKeyIsPressed;
    bool dKeyIsPressed;
    bool rKeyIsPressed;
    bool fKeyIsPressed;

    CameraRotateCenter *camera;
    double scroll = 0;

    float mouse_sensibility = 0.002;
    float mouse_move_speed = 0.00005;

    // Mouse information
    double last_xpos, last_ypos;
    bool drag = false;

public:
    explicit InputHandler(CameraRotateCenter *camera);

    void onKeyEvent(int key, int scancode, int action, int mods) override;

    void onMouseButtonEvent(int button, int action, int mods) override;

    void onCursorPositionChanged(double input_xPos, double yPos) override;

    void onScrollChanged(double xScrollOffset, double yScrollOffest) override;

};

