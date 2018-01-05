#pragma once

#include "WindowManager.hpp"

class InputHandler_I :
        public KeyEventListener,
        public MouseButtonEventListener,
        public CursorPositionListener,
        public ScrollListener
{
public:
    virtual void onKeyEvent(int key, int scancode, int action, int mods)=0;
    virtual void onMouseButtonEvent(int button, int action, int mods)=0;
    virtual void onCursorPositionChanged(double xPos, double yPos)=0;
    virtual void onScrollChanged(double xScrollOffset, double yScrollOffest)=0;
};
