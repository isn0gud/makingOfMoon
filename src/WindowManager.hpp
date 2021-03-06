#pragma once

#include "common.hpp"

static void keyCallbackFunction(GLFWwindow *window, int key, int scancode, int action, int mods);

static void windowCallbackFunction(GLFWwindow *window, int width, int height);

static void mouseButtonCallbackFunction(GLFWwindow *window, int button, int action, int mods);

static void cursorPositionCallbackFunction(GLFWwindow *window, double xpos, double ypos);

static void scrollCallbackFunction(GLFWwindow *window, double xScrollOffset, double yScrollOffest);

class KeyEventListener {
public:
    virtual void onKeyEvent(int key, int scancode, int action, int mods)=0;
};

class WindowEventListener {
public:
    virtual void onWindowSizeChanged(int width, int height)=0;
};

class MouseButtonEventListener {
public:
    virtual void onMouseButtonEvent(int button, int action, int mods)=0;
};

class CursorPositionListener {
public:
    virtual void onCursorPositionChanged(double xPos, double yPos)=0;
};

class ScrollListener {
public:
    virtual void onScrollChanged(double xScrollOffset, double yScrollOffest)=0;
};

class WindowManager {
    // Singleton class used to wrap glfw (To avoid mixing the C-style API with object oriented C++)
private:
    WindowManager();

    WindowManager(const WindowManager &);

    void operator=(const WindowManager &);

    static WindowManager *instance;

    int width;
    int height;
    GLFWwindow *window;
    std::vector<KeyEventListener *> keyEventListeners;
    std::vector<WindowEventListener *> windowEventListeners;
    std::vector<MouseButtonEventListener *> mouseButtonEventListeners;
    std::vector<CursorPositionListener *> cursorPositionListeners;
    std::vector<ScrollListener *> scrollListeners;

    friend void keyCallbackFunction(GLFWwindow *window, int key, int scancode, int action, int mods);

    friend void windowCallbackFunction(GLFWwindow *window, int width, int height);

    friend void mouseButtonCallbackFunction(GLFWwindow *window, int button, int action, int mods);

    friend void cursorPositionCallbackFunction(GLFWwindow *window, double xpos, double ypos);

    friend void scrollCallbackFunction(GLFWwindow *window, double xScrollOffset, double yScrollOffest);

public:
    static WindowManager *getInstance();

    void open(int width, int height, std::string title, bool vsync);

    void signalShouldClose();

    bool isOpen();

    bool shouldClose();

    void close();

    int getWidth();

    int getHeight();

    void setTitle(std::string title);

    void swapBuffers();

    void disableCursor();

    void enableCursor();
    //glm::vec2 getMousePosition();

    void addKeyEventListener(KeyEventListener *listener);

    void removeKeyEventListener(KeyEventListener *listener);

    void addWindowEventListener(WindowEventListener *listener);

    void removeWindowEventListener(WindowEventListener *listener);

    void addMouseButtonEventListener(MouseButtonEventListener *listener);

    void removeMouseButtonEvenetListener(MouseButtonEventListener *listener);

    void addCursorPositionListener(CursorPositionListener *listener);

    void removeCursorPositionListener(CursorPositionListener *listener);

    void addScrollListener(ScrollListener *listener);

    void removeScrollListener(ScrollListener *listener);
};
