#include "WindowManager.hpp"
#include <iostream>

static void errorCallbackFunction(int error, const char *description) {
    std::cerr << "GLFW-Error: " << description << std::endl;
}

static void keyCallbackFunction(GLFWwindow *window, int key, int scancode, int action, int mods) {
    WindowManager *wm = WindowManager::getInstance();
    for (auto &keyEventListener : wm->keyEventListeners)
        keyEventListener->onKeyEvent(key, scancode, action, mods);
}

static void windowCallbackFunction(GLFWwindow *window, int width, int height) {
    WindowManager *wm = WindowManager::getInstance();
    for (auto &windowEventListener : wm->windowEventListeners)
        windowEventListener->onWindowSizeChanged(width, height);
}

static void mouseButtonCallbackFunction(GLFWwindow *window, int button, int action, int mods) {
    WindowManager *wm = WindowManager::getInstance();
    for (auto &mouseButtonEventListener : wm->mouseButtonEventListeners)
        mouseButtonEventListener->onMouseButtonEvent(button, action, mods);
}

static void cursorPositionCallbackFunction(GLFWwindow *window, double xpos, double ypos) {
    WindowManager *wm = WindowManager::getInstance();
    for (auto &cursorPositionListener : wm->cursorPositionListeners)
        cursorPositionListener->onCursorPositionChanged(xpos, ypos);

}

static void scrollCallbackFunction(GLFWwindow *window, double xScrollOffset, double yScrollOffest) {
    WindowManager *wm = WindowManager::getInstance();
    for (auto &scrollListener : wm->scrollListeners)
        scrollListener->onScrollChanged(xScrollOffset, yScrollOffest);
}

WindowManager::WindowManager() {
    width = 0;
    height = 0;
    window = nullptr;
}

WindowManager *WindowManager::instance = nullptr;

WindowManager *WindowManager::getInstance() {
    if (!instance)
        instance = new WindowManager();
    return instance;
}

void WindowManager::open(int width, int height, std::string title, bool vsync) {
    glfwSetErrorCallback(errorCallbackFunction);
    if (!glfwInit()) throw std::runtime_error("glfw init failed");

    this->width = width;
    this->height = height;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Can't open glfw window");
    }
    glfwMakeContextCurrent(window);

    glfwSetKeyCallback(window, keyCallbackFunction);
    glfwSetWindowSizeCallback(window, windowCallbackFunction);
    glfwSetMouseButtonCallback(window, mouseButtonCallbackFunction);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPosCallback(window, cursorPositionCallbackFunction);
    glfwSetScrollCallback(window, scrollCallbackFunction);

    glewExperimental = true;
    GLenum error = glewInit();
    if (error != GLEW_OK) {
        throw std::runtime_error("Can't load GL");
    }
}

void WindowManager::signalShouldClose() {
    glfwSetWindowShouldClose(window, 1);
}

void WindowManager::close() {
    glfwDestroyWindow(window);
    glfwTerminate();
    window = nullptr;
}

void WindowManager::swapBuffers() {
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void WindowManager::addKeyEventListener(KeyEventListener *listener) {
    keyEventListeners.push_back(listener);
}

void WindowManager::removeKeyEventListener(KeyEventListener *listener) {
    for (int i = 0; i < keyEventListeners.size(); i++) {
        if (listener == keyEventListeners[i])
            keyEventListeners.erase(keyEventListeners.begin() + i);
        break;
    }
}

void WindowManager::addWindowEventListener(WindowEventListener *listener) {
    windowEventListeners.push_back(listener);
}

void WindowManager::removeWindowEventListener(WindowEventListener *listener) {
    for (int i = 0; i < windowEventListeners.size(); i++) {
        if (listener == windowEventListeners[i])
            windowEventListeners.erase(windowEventListeners.begin() + i);
        break;
    }
}

void WindowManager::addMouseButtonEventListener(MouseButtonEventListener *listener) {
    mouseButtonEventListeners.push_back(listener);
}

void WindowManager::removeMouseButtonEvenetListener(MouseButtonEventListener *listener) {
    for (int i = 0; i < mouseButtonEventListeners.size(); i++) {
        if (listener == mouseButtonEventListeners[i])
            mouseButtonEventListeners.erase(mouseButtonEventListeners.begin() + i);
        break;
    }
}

void WindowManager::addCursorPositionListener(CursorPositionListener *listener) {
    cursorPositionListeners.push_back(listener);
}

void WindowManager::removeCursorPositionListener(CursorPositionListener *listener) {
    for (int i = 0; i < cursorPositionListeners.size(); i++) {
        if (listener == cursorPositionListeners[i])
            cursorPositionListeners.erase(cursorPositionListeners.begin() + i);
        break;
    }
}


void WindowManager::addScrollListener(ScrollListener *listener) {
    scrollListeners.push_back(listener);
}

void WindowManager::removeScrollListener(ScrollListener *listener) {
    for (int i = 0; i < scrollListeners.size(); i++) {
        if (listener == scrollListeners[i])
            scrollListeners.erase(scrollListeners.begin() + i);
        break;
    }
}


int WindowManager::getWidth() {
    return width;
}

int WindowManager::getHeight() {
    return height;
}

bool WindowManager::isOpen() {
    return (window != nullptr);
}

bool WindowManager::shouldClose() {
    return (bool) glfwWindowShouldClose(window);
}

void WindowManager::setTitle(std::string title) {
    glfwSetWindowTitle(window, title.c_str());
}

void WindowManager::disableCursor() {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void WindowManager::enableCursor() {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}


