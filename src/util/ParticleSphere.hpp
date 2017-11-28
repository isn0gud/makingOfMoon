//
// Created by Pius Friesch on 20/11/2017.
//

#ifndef ASS_OPENGL_PARTICLESPHERE_HPP
#define ASS_OPENGL_PARTICLESPHERE_HPP


#include "Sphere.hpp"

class ParticleSphere : public Sphere {

#define VELO_TO_COLOR_SCALING 10.0
#define DEFAULT_SLICES 15
#define DEFAULT_STACKS 15
#define SPHERE_SIZE 0.05f


public:
//    void setAlpha(float alpha) override;

    glm::mat4 getModel() const override;

    glm::vec4 getColor() const override;

    GLfloat getRadius() const override;

    GLint getSlices() const override;

    GLint getStacks() const override;

//    ParticleSphere(Particle *particle,  GLint slices, GLint stacks);

    ParticleSphere(Particle *particle);


private:

    Particle *particle;
    float alpha = 0.2f;
//    GLfloat radius;
    GLint slices;
    GLint stacks;

};


#endif //ASS_OPENGL_PARTICLESPHERE_HPP
