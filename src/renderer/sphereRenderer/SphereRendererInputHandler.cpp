#include "SphereRendererInputHandler.hpp"

void SphereRendererInputHandler::onKeyEvent(int key, int scancode, int action, int mods) {
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

void SphereRendererInputHandler::onMouseButtonEvent(int button, int action, int mods) {
    if (action == GLFW_PRESS) {
        mouseIsPressed = true;
        WindowManager::getInstance()->disableCursor();
    } else if (action == GLFW_RELEASE) {
        mouseIsPressed = false;
        WindowManager::getInstance()->enableCursor();
    }
}

void SphereRendererInputHandler::onCursorPositionChanged(double xPos, double yPos) {
    glm::vec2 lastMousePosition = mousePosition;
    mousePosition = glm::vec2((float) xPos, (float) yPos);

    if (mouseIsPressed)
        mouseMovement += mousePosition - lastMousePosition;
}

void SphereRendererInputHandler::onScrollChanged(double xScrollOffset, double yScrollOffest)
{

}

const ShpereRendererInputDerivedData &SphereRendererInputHandler::getDerivedData() {
    data.cameraLocalVelocity = glm::vec3(0);
    if (wKeyIsPressed)
        data.cameraLocalVelocity.z += -1;
    if (sKeyIsPressed)
        data.cameraLocalVelocity.z += 1;
    if (aKeyIsPressed)
        data.cameraLocalVelocity.x += -1;
    if (dKeyIsPressed)
        data.cameraLocalVelocity.x += 1;
    if (rKeyIsPressed)
        data.cameraLocalVelocity.y += 1;
    if (fKeyIsPressed)
        data.cameraLocalVelocity.y += -1;

    data.mouseMovement = mouseMovement;
    mouseMovement = glm::vec2(0);

    return data;
}
