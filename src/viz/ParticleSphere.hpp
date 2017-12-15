//
// Created by Pius Friesch on 20/11/2017.
//

#ifndef AGP_PROJECT_PARTICLESPHERE_HPP
#define AGP_PROJECT_PARTICLESPHERE_HPP


#include "Sphere.hpp"

class ParticleSphere : public Sphere {

#define VELO_TO_COLOR_SCALING 10.0
#define DEFAULT_SLICES 15
#define DEFAULT_STACKS 15
#define SCALING 1.0f

public:
    glm::mat4 getTransformationMatrix() const override;
    glm::vec4 getColor() const override;
    GLfloat getRadius() const override;

    ParticleSphere(Particle *particle);

private:

    Particle *particle;
    float alpha = 1.f;
};


#endif //AGP_PROJECT_PARTICLESPHERE_HPP
