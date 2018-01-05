#include "SpriteRendererInputHandler.hpp"

void SpriteRendererInputHandler::onKeyEvent(int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_W)
            wKeyIsPressed = true;
        else if (key == GLFW_KEY_A)
            aKeyIsPressed = true;
        else if (key == GLFW_KEY_S)
            sKeyIsPressed = true;
        else if (key == GLFW_KEY_D)
            dKeyIsPressed = true;
        else if (key == GLFW_KEY_R)
            rKeyIsPressed = true;
        else if (key == GLFW_KEY_F)
            fKeyIsPressed = true;
    } else if (action == GLFW_RELEASE) {
        if (key == GLFW_KEY_W)
            wKeyIsPressed = false;
        else if (key == GLFW_KEY_A)
            aKeyIsPressed = false;
        else if (key == GLFW_KEY_S)
            sKeyIsPressed = false;
        else if (key == GLFW_KEY_D)
            dKeyIsPressed = false;
        else if (key == GLFW_KEY_R)
            rKeyIsPressed = false;
        else if (key == GLFW_KEY_F)
            fKeyIsPressed = false;
    }
}

void SpriteRendererInputHandler::onMouseButtonEvent(int button, int action, int mods) {
    if (action == GLFW_PRESS) {
        mouseIsPressed = true;
        WindowManager::getInstance()->disableCursor();
    } else if (action == GLFW_RELEASE) {
        mouseIsPressed = false;
        WindowManager::getInstance()->enableCursor();
    }
}

void SpriteRendererInputHandler::onCursorPositionChanged(double input_xPos, double input_yPos) {
// View movement
    if (mouseIsPressed) {
        if (!drag) {
            last_xpos = input_xPos;
            last_ypos = input_yPos;
            drag = true;
        }
        double xpos, ypos;
        double xdiff, ydiff;
        xpos = input_xPos;
        ypos = input_yPos;
        xdiff = xpos - last_xpos;
        ydiff = ypos - last_ypos;

        last_xpos = xpos;
        last_ypos = ypos;

        camera->addVelocity(glm::vec3(xdiff, -ydiff, 0) * mouse_sensibility);
    } else drag = false;

}

void SpriteRendererInputHandler::onScrollChanged(double xScrollOffset, double yScrollOffest) {
    scroll += yScrollOffest;
    // Scrolling
    camera->addVelocity(glm::vec3(0, 0, scroll * 0.02));
    scroll = 0;
}

SpriteRendererInputHandler::SpriteRendererInputHandler(CameraRotateCenter *camera) : camera(camera) {
    mouseIsPressed = false;
    wKeyIsPressed = false;
    aKeyIsPressed = false;
    sKeyIsPressed = false;
    dKeyIsPressed = false;
    rKeyIsPressed = false;
    fKeyIsPressed = false;
}



