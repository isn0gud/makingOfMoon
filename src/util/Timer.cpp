#include "Timer.hpp"

void Timer::start()
{
    startTime = std::chrono::system_clock::now();
    lastFrameTime = startTime;
}

float Timer::getTimeSinceStart()
{
    return std::chrono::duration_cast< std::chrono::duration<float> >(std::chrono::system_clock::now() - startTime).count();
}

float Timer::getFrameTime()
{
    auto newFrameTime = std::chrono::system_clock::now();
    float result = (std::chrono::duration_cast< std::chrono::duration<float> >(newFrameTime - lastFrameTime)).count();
    lastFrameTime = newFrameTime;
    return result;
}
