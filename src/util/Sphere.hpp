//
// Created by Pius Friesch on 18/11/2017.
//

#ifndef ASS_OPENGL_SPHERE_H
#define ASS_OPENGL_SPHERE_H

#include "../common.hpp"
#include "../sim/Particle.hpp"

/**
 * Dataclass to save to properties of a Sphere.
 */
class Sphere {


public:
    enum Shape {
        WIRE,
        SOLID
    };

//    virtual void setAlpha(float alpha) = 0;

    virtual glm::mat4 getModel() const = 0;

    virtual glm::vec4 getColor() const = 0;

    virtual GLfloat getRadius() const = 0;

    virtual GLint getSlices() const = 0;

    virtual GLint getStacks() const = 0;


};


#endif //ASS_OPENGL_SPHERE_H
