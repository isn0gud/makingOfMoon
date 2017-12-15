//
// Created by friesch on 12/15/17.
//

#ifndef MAKINGOFMOON_PARTICLES_H
#define MAKINGOFMOON_PARTICLES_H

#include "../common.hpp"

class Particles {

public:


    int num_particles = 0;

    enum TYPE {
        IRON,
        SILICATE
    };

    TYPE *type = nullptr;

    //unit: km
    glm::vec4 *particlePos = nullptr; //TODO: can't use a std::vector right now since the gl buffer mapping to cpu space gives an allocated pointer already, which has to be used.
    //unit: km/s
    glm::vec4 *particleVelo = nullptr;
    //unit: km/s^2
    glm::vec4 *particleAccel = nullptr;
    GLfloat *radius = nullptr;

    //unit: kg
    GLfloat *mass = nullptr;

    GLfloat *shellDepthFraction = nullptr; //P: SDP
    GLfloat *&SDP = shellDepthFraction;

    //unit: kg /(m*s^2)
    GLfloat *elasticSpringConstant = nullptr; //P: K
    GLfloat *&K = elasticSpringConstant;

    GLfloat *inelasticSpringForceReductionFactor = nullptr; //P: KRP
    GLfloat *&KRP = inelasticSpringForceReductionFactor;

};

#endif //MAKINGOFMOON_PARTICLES_H
