//
// Created by Pius Friesch on 19/11/2017.
//

#ifndef ASS_OPENGL_PARTICLE_HPP
#define ASS_OPENGL_PARTICLE_HPP

#include "../common.hpp"

class Particle {


public:

    glm::vec3 pos;
    glm::vec3 velo;
    glm::vec3 accel;

    GLfloat radius;
    GLfloat mass;

    GLfloat shellDepthFraction;
    GLfloat elasticSpringConstant;
    GLfloat inelasticSpringForceReductionFactor;

    Particle(glm::vec3 pos, glm::vec3 velo, glm::vec3 accel, GLfloat radius) : pos(pos),
                                                                               velo(velo),
                                                                               accel(accel),
                                                                               radius(radius),
                                                                               shellDepthFraction(0.5),
                                                                               mass(1),
                                                                               elasticSpringConstant(1),
                                                                               inelasticSpringForceReductionFactor(1) {}

    Particle(glm::vec3 pos, glm::vec3 velo, glm::vec3 accel, GLfloat radius, GLfloat mass) : pos(pos),
                                                                                             velo(velo),
                                                                                             accel(accel),
                                                                                             radius(radius),
                                                                                             shellDepthFraction(0.5),
                                                                                             mass(mass),
                                                                                             elasticSpringConstant(1),
                                                                                             inelasticSpringForceReductionFactor(
                                                                                                     1) {}

};

#endif //ASS_OPENGL_PARTICLE_HPP
