//
// Created by Pius Friesch on 20/11/2017.
//

#include "StaticSphere.hpp"

//void StaticSphere::setAlpha(float alpha) {
//
//    color.a = alpha;
//
//}

glm::mat4 StaticSphere::getModel() const {
    return model;
}

glm::vec4 StaticSphere::getColor() const {
    return color;
}

GLfloat StaticSphere::getRadius() const {
    return radius;
}

GLint StaticSphere::getSlices() const {
    return slices;
}

GLint StaticSphere::getStacks() const {
    return stacks;
}

StaticSphere::StaticSphere(const glm::vec3 &pos, const glm::vec4 &color, GLfloat radius, GLint slices, GLint stacks)
        : color(color), radius(radius), slices(slices), stacks(stacks) { model = glm::translate(glm::mat4(1), pos); }

StaticSphere::StaticSphere(const glm::vec3 &pos, const glm::vec4 &color, GLfloat radius)
        : color(color), radius(radius), slices(10), stacks(10) { model = glm::translate(glm::mat4(1), pos); }