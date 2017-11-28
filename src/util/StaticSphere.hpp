//
// Created by Pius Friesch on 20/11/2017.
//

#ifndef ASS_OPENGL_STATICSPHERE_HPP
#define ASS_OPENGL_STATICSPHERE_HPP


#include "Sphere.hpp"

class StaticSphere : public Sphere {
public:
//    void setAlpha(float alpha) override;

    glm::mat4 getModel() const override;

    glm::vec4 getColor() const override;

    GLfloat getRadius() const override;

    GLint getSlices() const override;

    GLint getStacks() const override;

    StaticSphere(const glm::vec3 &pos, const glm::vec4 &color, GLfloat radius, GLint slices, GLint stacks);

    StaticSphere(const glm::vec3 &pos, const glm::vec4 &color, GLfloat radius);

private:
    glm::mat4 model;
    glm::vec4 color;
    GLfloat radius;
    GLint slices;
    GLint stacks;
};


#endif //ASS_OPENGL_STATICSPHERE_HPP
