#include "InputHandler.hpp"

void InputHandler::onKeyEvent(int key, int scancode, int action, int mods) {
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

void InputHandler::onMouseButtonEvent(int button, int action, int mods) {
    if (action == GLFW_PRESS) {
        mouseIsPressed = true;
        WindowManager::getInstance()->disableCursor();
    } else if (action == GLFW_RELEASE) {
        mouseIsPressed = false;
        WindowManager::getInstance()->enableCursor();
    }
}

void InputHandler::onCursorPositionChanged(double input_xPos, double input_yPos) {
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

//const RendererInputDerivedData &InputHandler::getDerivedData() {
//    data.cameraLocalVelocity = glm::vec3(0);
//    if (wKeyIsPressed)
//        data.cameraLocalVelocity.z += -1;
//    if (sKeyIsPressed)
//        data.cameraLocalVelocity.z += 1;
//    if (aKeyIsPressed)
//        data.cameraLocalVelocity.x += -1;
//    if (dKeyIsPressed)
//        data.cameraLocalVelocity.x += 1;
//    if (rKeyIsPressed)
//        data.cameraLocalVelocity.y += 1;
//    if (fKeyIsPressed)
//        data.cameraLocalVelocity.y += -1;
//
//    data.mouseMovement = mouseMovement;
//    mouseMovement = glm::vec2(0);
//
//    return data;
//}

void InputHandler::onScrollChanged(double xScrollOffset, double yScrollOffest) {
    scroll += yScrollOffest;
    // Scrolling
    camera->addVelocity(glm::vec3(0, 0, scroll * 0.02));
    scroll = 0;
}

InputHandler::InputHandler(CameraRotateCenter *camera) : camera(camera) {
    mouseIsPressed = false;
    wKeyIsPressed = false;
    aKeyIsPressed = false;
    sKeyIsPressed = false;
    dKeyIsPressed = false;
    rKeyIsPressed = false;
    fKeyIsPressed = false;
}



