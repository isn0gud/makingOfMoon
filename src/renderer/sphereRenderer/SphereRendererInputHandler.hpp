#pragma once

#include "../../common.hpp"
#include "../../InputHandler_I.hpp"

struct ShpereRendererInputDerivedData {
    glm::vec3 cameraLocalVelocity;
    glm::vec2 mouseMovement;
};

// class that handles the user input of the renderer and makes it available in a more easily processed format
class SphereRendererInputHandler : public InputHandler_I {
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

    ShpereRendererInputDerivedData data;

public:
    SphereRendererInputHandler() {
        mouseIsPressed = false;
        wKeyIsPressed = false;
        aKeyIsPressed = false;
        sKeyIsPressed = false;
        dKeyIsPressed = false;
        rKeyIsPressed = false;
        fKeyIsPressed = false;
        data.mouseMovement = glm::vec2(0);
        data.cameraLocalVelocity = glm::vec3(0);

        mousePosition = glm::vec2(0);
        mouseMovement = glm::vec2(0);
    }

    void onKeyEvent(int key, int scancode, int action, int mods) override;

    void onMouseButtonEvent(int button, int action, int mods) override;

    void onCursorPositionChanged(double xPos, double yPos) override;

    void onScrollChanged(double xScrollOffset, double yScrollOffest) override;

    const ShpereRendererInputDerivedData &getDerivedData();
};
