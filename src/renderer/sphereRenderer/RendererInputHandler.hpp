#pragma once

#include "../../common.hpp"

struct RendererInputDerivedData {
    glm::vec3 cameraLocalVelocity;
    glm::vec2 mouseMovement;
};

// class that handles the user input of the renderer and makes it available in a more easily processed format
class RendererInputHandler : public KeyEventListener, public MouseButtonEventListener, public CursorPositionListener {
private:
    bool mouseIsPressed;
    bool wKeyIsPressed;
    bool aKeyIsPressed;
    bool sKeyIsPressed;
    bool dKeyIsPressed;
    bool rKeyIsPressed;
    bool fKeyIsPressed;

    glm::vec2 mouseMovement;
    glm::vec2 mousePosition;

    RendererInputDerivedData data;

public:
    RendererInputHandler() {
        mouseIsPressed = false;
        wKeyIsPressed = false;
        aKeyIsPressed = false;
        sKeyIsPressed = false;
        dKeyIsPressed = false;
        rKeyIsPressed = false;
        fKeyIsPressed = false;
        data.mouseMovement = glm::vec2(0);
        data.cameraLocalVelocity = glm::vec3(0);
    }

    void onKeyEvent(int key, int scancode, int action, int mods);

    void onMouseButtonEvent(int button, int action, int mods);

    void onCursorPositionChanged(double xPos, double yPos);

    const RendererInputDerivedData &getDerivedData();
};
