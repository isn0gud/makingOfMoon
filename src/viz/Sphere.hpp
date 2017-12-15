//
// Created by Pius Friesch on 18/11/2017.
//

#ifndef AGP_PROJECT_SPHERE_HPP
#define AGP_PROJECT_SPHERE_HPP

#include "../common.hpp"
#include "../sim/Particle.hpp"

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


#endif //AGP_PROJECT_SPHERE_HPP
