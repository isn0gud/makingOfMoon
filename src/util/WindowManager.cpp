#include "WindowManager.hpp"
#include <iostream>

using namespace std;

static void errorCallbackFunction(int error, const char* description)
{
    cerr << "GLFW-Error: " << description << endl;
}

static void keyCallbackFunction(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    WindowManager* wm = WindowManager::getInstance();
    for(int i = 0; i < wm->keyEventListeners.size(); i++)
        wm->keyEventListeners[i]->onKeyEvent(key, scancode, action, mods);
}

static void windowCallbackFunction(GLFWwindow *window, int width, int height)
{
    WindowManager* wm = WindowManager::getInstance();
    for(int i = 0; i < wm->windowEventListeners.size(); i++)
        wm->windowEventListeners[i]->onWindowSizeChanged(width, height);
}

static void mouseButtonCallbackFunction(GLFWwindow* window, int button, int action, int mods)
{
    WindowManager* wm = WindowManager::getInstance();
    for(int i = 0; i < wm->mouseButtonEventListeners.size(); i++)
        wm->mouseButtonEventListeners[i]->onMouseButtonEvent(button, action, mods);
}

static void cursorPositionCallbackFunction(GLFWwindow *window, double xpos, double ypos)
{
    WindowManager* wm = WindowManager::getInstance();
    for(int i = 0; i < wm->cursorPositionListeners.size(); i++)
        wm->cursorPositionListeners[i]->onCursorPositionChanged(xpos, ypos);

}

WindowManager::WindowManager()
{
    width = 0;
    height = 0;
    window = NULL;
}

WindowManager* WindowManager::instance = 0;

WindowManager* WindowManager::getInstance()
{
    if (!instance)
        instance = new WindowManager();
    return instance;
}

bool WindowManager::open(int width, int height, string title, bool vsync)
{
    glfwSetErrorCallback(errorCallbackFunction);
    int returnCode = glfwInit();
    if(!returnCode)
        return true;

    this->width = width;
    this->height = height;



    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return true;
    }

    glfwSetKeyCallback(window, keyCallbackFunction);
    glfwSetWindowSizeCallback(window, windowCallbackFunction);
    glfwSetMouseButtonCallback(window, mouseButtonCallbackFunction);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPosCallback(window, cursorPositionCallbackFunction);

    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
    if(vsync)
        glfwSwapInterval(1);

    return false;
}

void WindowManager::singalShouldClose()
{
    glfwSetWindowShouldClose(window, 1);
}

void WindowManager::close()
{
    glfwDestroyWindow(window);
    glfwTerminate();
    window = NULL;
}

void WindowManager::swapBuffers()
{
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void WindowManager::addKeyEventListener(KeyEventListener* listener)
{
    keyEventListeners.push_back(listener);
}

void WindowManager::removeKeyEventListener(KeyEventListener* listener)
{
    for(int i = 0; i < keyEventListeners.size(); i++)
    {
        if(listener == keyEventListeners[i])
            keyEventListeners.erase(keyEventListeners.begin() + i);
        break;
    }
}

void WindowManager::addWindowEventListener(WindowEventListener* listener)
{
    windowEventListeners.push_back(listener);
}

void WindowManager::removeWindowEventListener(WindowEventListener* listener)
{
    for(int i = 0; i < windowEventListeners.size(); i++)
    {
        if(listener == windowEventListeners[i])
            windowEventListeners.erase(windowEventListeners.begin() + i);
        break;
    }
}

void WindowManager::addMouseButtonEventListener(MouseButtonEventListener *listener)
{
    mouseButtonEventListeners.push_back(listener);
}

void WindowManager::removeMouseButtonEvenetListener(MouseButtonEventListener *listener)
{
    for(int i = 0; i < mouseButtonEventListeners.size(); i++)
    {
        if(listener == mouseButtonEventListeners[i])
            mouseButtonEventListeners.erase(mouseButtonEventListeners.begin() + i);
        break;
    }
}

void WindowManager::addCursorPositionListener(CursorPositionListener *listener)
{
    cursorPositionListeners.push_back(listener);
}

void WindowManager::removeCursorPositionListener(CursorPositionListener *listener)
{
    for(int i = 0; i < cursorPositionListeners.size(); i++)
    {
        if(listener == cursorPositionListeners[i])
            cursorPositionListeners.erase(cursorPositionListeners.begin() + i);
        break;
    }
}

int WindowManager::getWidth()
{
    return width;
}

int WindowManager::getHeight()
{
    return height;
}

bool WindowManager::isOpen()
{
    return (window != NULL);
}

bool WindowManager::shouldClose()
{
    return (bool)glfwWindowShouldClose(window);
}

void WindowManager::setTitle(string title)
{
    glfwSetWindowTitle(window, title.c_str());
}

void WindowManager::disableCursor()
{
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void WindowManager::enableCursor()
{
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}
