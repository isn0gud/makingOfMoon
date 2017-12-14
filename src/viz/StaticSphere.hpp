//
// Created by Pius Friesch on 20/11/2017.
//

#ifndef AGP_PROJECT_STATICSPHERE_HPP
#define AGP_PROJECT_STATICSPHERE_HPP

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


#endif //AGP_PROJECT_STATICSPHERE_HPP
