#pragma once

#include "../../common.hpp"

/**
 * Dataclass to save to properties of a Sphere.
 */
class Sphere {


public:
    enum Shape {
        WIRE,
        SOLID,
        WIRE_AND_SOLID
    };

    virtual glm::mat4 getTransformationMatrix() const = 0;

    virtual glm::vec4 getColor() const = 0;

    virtual GLfloat getRadius() const = 0;
};


