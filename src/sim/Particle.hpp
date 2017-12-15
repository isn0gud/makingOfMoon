//
// Created by Pius Friesch on 19/11/2017.
//

#ifndef AGP_PROJECT_PARTICLE_HPP
#define AGP_PROJECT_PARTICLE_HPP

#include "../common.hpp"

class Particle {


public:

    enum TYPE {
        IRON,
        SILICATE
    };

    const TYPE type;

    //unit: km
    glm::vec3 pos;
    //unit: km/s
    glm::vec3 velo;
    //unit: km/s^2
    glm::vec3 accel;

    //unit: km
    const GLfloat radius;

    const GLfloat diameter() const { return 2 * radius; }

    //unit: kg
    const GLfloat mass;

    const GLfloat shellDepthFraction; //P: SDP
    const GLfloat &SDP = shellDepthFraction;

    //unit: kg /(m*s^2)
    const GLfloat elasticSpringConstant; //P: K
    const GLfloat &K = elasticSpringConstant;

    const GLfloat inelasticSpringForceReductionFactor; //P: KRP
    const GLfloat &KRP = inelasticSpringForceReductionFactor;


    Particle(const Particle &p2) : pos(p2.pos),
                                   velo(p2.velo),
                                   accel(p2.accel),
                                   radius(p2.radius),
                                   mass(p2.mass),
                                   shellDepthFraction(p2.shellDepthFraction),
                                   elasticSpringConstant(p2.elasticSpringConstant),
                                   inelasticSpringForceReductionFactor(p2.inelasticSpringForceReductionFactor),
                                   type(p2.type) {

    }

    Particle(const glm::vec3 &pos,
             const glm::vec3 &velo,
             const glm::vec3 &accel,
             const GLfloat radius,
             const GLfloat mass,
             const GLfloat shellDepthFraction,
             const GLfloat elasticSpringConstant,
             const GLfloat inelasticSpringForceReductionFactor,
             const Particle::TYPE type)
            : pos(pos),
              velo(velo),
              accel(accel),
              radius(radius),
              mass(mass),
              shellDepthFraction(shellDepthFraction),
              elasticSpringConstant(elasticSpringConstant),
              inelasticSpringForceReductionFactor(inelasticSpringForceReductionFactor),
              type(type) {}


    static Particle *particleFromType(TYPE type, float radius) {
        float mass = 0;
        float SDF = 0;
        float K = 0;
        float KRF = 0;

        switch(type)
        {
        case Particle::TYPE::SILICATE:
            mass = 7.4161E19 * pow(radius / 188.39, 3);
            SDF = 1 - 0.001;
            K = 7.2785E10; // TODO: Should probably scale with radius somehow
            KRF = 0.01;
            break;
        case Particle::TYPE::IRON:
            mass = 1.9549E20 * pow(radius / 188.39, 3);
            SDF = 1 - 0.01;
            K = 2.9114E11; // TODO: Should probably scale with radius somehow
            KRF = 0.02;
            break;
        }

        return new Particle(glm::vec3(0), glm::vec3(0), glm::vec3(0), radius, mass, SDF, K, KRF, type);
    }

// REAL VALUES:


//    static Particle *silicateParticle(glm::vec3 pos) {
//        return new Particle(pos, glm::vec3(0), glm::vec3(0), 188.39, 7.4161E19, 0.001, 7.2785E10, 0.01,
//                            Particle::TYPE::SILICATE);
//
//    }
//
//    static Particle *ironParticle(glm::vec3 pos) {
//
//        //D: 376.78 /2 = 188.39
//        //SDP_Fe of D = 0.01 / 2
//        //K_Fe = 2.9114E11 =
//        return new Particle(pos, glm::vec3(0), glm::vec3(0), 188.39, 1.9549E20, 0.01, 2.9114E11, 0.02,
//                            Particle::TYPE::IRON);
//    }

};


#endif //AGP_PROJECT_PARTICLE_HPP
