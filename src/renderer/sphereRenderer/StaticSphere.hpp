#pragma once

#include "Sphere.hpp"

class StaticSphere : public Sphere {
public:
    glm::mat4 getTransformationMatrix() const override;

    glm::vec4 getColor() const override;

    GLfloat getRadius() const override;

    StaticSphere(const glm::vec3 &pos, const glm::vec4 &color, GLfloat radius);

private:
    glm::mat4 transformationMatrix;
    glm::vec4 color;
    GLfloat radius;
};
