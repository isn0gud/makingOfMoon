//
// Created by Pius Friesch on 20/11/2017.
//

#include "ParticleSphere.hpp"

glm::mat4 ParticleSphere::getTransformationMatrix() const {
    return glm::translate(glm::mat4(1), (particle->pos * SCALING));
}

glm::vec4 ParticleSphere::getColor() const {
    switch (particle->type) {
        case Particle::TYPE::IRON:
            return glm::vec4(0.8f, 0.1f, 0.1f, alpha);
        case Particle::TYPE::SILICATE:
            return glm::vec4(0.8f, 0.8f, 0.1f, alpha);
        default:
            return glm::vec4(0.8f, 0.8f, 0.8f, alpha);
    }
}

GLfloat ParticleSphere::getRadius() const {
    return particle->radius * (GLfloat) SCALING;
}

ParticleSphere::ParticleSphere(Particle *particle) : particle(particle) {}

