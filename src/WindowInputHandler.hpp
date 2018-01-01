#pragma once

#include "WindowManager.hpp"

class WindowInputHandler : public KeyEventListener {
public:
    bool runSimulation;

    WindowInputHandler() : runSimulation(false) {}

    void onKeyEvent(int key, int scancode, int action, int mods) override {
        if (action == GLFW_PRESS) {
            if (key == GLFW_KEY_ESCAPE)
                WindowManager::getInstance()->signalShouldClose();
            else if (key == GLFW_KEY_SPACE)
                runSimulation = !runSimulation;
        }
    }
};
