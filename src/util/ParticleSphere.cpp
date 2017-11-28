//
// Created by Pius Friesch on 20/11/2017.
//

#include "ParticleSphere.hpp"

//void ParticleSphere::setAlpha(float alpha) {
//    this->alpha = alpha;
//}


glm::mat4 ParticleSphere::getModel() const {
    return glm::translate(glm::mat4(1), particle->pos);
}

glm::vec4 ParticleSphere::getColor() const {
//    return glm::vec4(0.1f, 0.1f, 0.4f, alpha);

    return glm::vec4(glm::vec3(particle->mass), alpha);
}


GLfloat ParticleSphere::getRadius() const {
    return particle->radius;
}

GLint ParticleSphere::getSlices() const {
    return slices;
}

GLint ParticleSphere::getStacks() const {
    return stacks;
}

//ParticleSphere::ParticleSphere(Particle *particle, GLint slices, GLint stacks) : particle(particle),
//                                                                                 slices(slices),
//                                                                                 stacks(stacks) {}

ParticleSphere::ParticleSphere(Particle *particle) : particle(particle),
                                                     slices(DEFAULT_SLICES),
                                                     stacks(DEFAULT_STACKS) {}

