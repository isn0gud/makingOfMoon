//
// Created by Pius Friesch on 20/11/2017.
//

#include "StaticSphere.hpp"

glm::mat4 StaticSphere::getTransformationMatrix() const {
    return transformationMatrix;
}

glm::vec4 StaticSphere::getColor() const {
    return color;
}

GLfloat StaticSphere::getRadius() const {
    return radius;
}

StaticSphere::StaticSphere(const glm::vec3 &pos, const glm::vec4 &color, GLfloat radius)
        : color(color), radius(radius) { transformationMatrix = glm::translate(glm::mat4(1), pos); }
