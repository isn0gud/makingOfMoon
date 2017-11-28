#ifndef ASS_OPENGL_TIMER_H
#define ASS_OPENGL_TIMER_H

#include "../common.hpp"

class Timer
{
private:
    std::chrono::time_point<std::chrono::system_clock> startTime;
    std::chrono::time_point<std::chrono::system_clock> lastFrameTime;
public:
    Timer() {}

    void start();
    float getTimeSinceStart();
    float getFrameTime();

};

#endif // ASS_OPENGL_TIMER_H
