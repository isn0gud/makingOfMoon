#pragma once

#include <cuda_runtime_api.h>
#include "common.hpp"


class Particles {


public:
    enum TYPE {
        IRON,
        SILICATE
    };

    explicit Particles(int numParticles) : numParticles(numParticles) {
        type = new TYPE[numParticles];
        pos = new glm::vec4[numParticles];
        velo = new glm::vec4[numParticles];
        accel = new glm::vec4[numParticles];
        radius = new GLfloat[numParticles];
        mass = new GLfloat[numParticles];
        shellDepthFraction = new GLfloat[numParticles];
        elasticSpringConstant = new GLfloat[numParticles];
        inelasticSpringForceReductionFactor = new GLfloat[numParticles];

    }

    typedef struct Particles_cuda {
        TYPE *type;
//        glm::vec4 *pos;
        glm::vec4 *velo;
        glm::vec4 *accel;

        GLfloat *radius;
        GLfloat *mass;
        GLfloat *shellDepthFraction;
        GLfloat *elasticSpringConstant;
        GLfloat *inelasticSpringForceReductionFactor;

        int numParticles;
    } Particles_cuda;

    Particles_cuda *to_cuda() {
        TYPE *type;
//        glm::vec4 *pos, *velo, *accel;
        glm::vec4 *velo, *accel;

        GLfloat *radius, *mass, *shellDepthFraction, *elasticSpringConstant, *inelasticSpringForceReductionFactor;
//        int *_numParticles;

        // Allocate device data
        cudaMalloc((void **) &type, numParticles * sizeof(*type));
//        cudaMalloc((void **) &pos, numParticles * sizeof(*pos));
        cudaMalloc((void **) &velo, numParticles * sizeof(*velo));
        cudaMalloc((void **) &accel, numParticles * sizeof(*accel));
        cudaMalloc((void **) &radius, numParticles * sizeof(*radius));
        cudaMalloc((void **) &mass, numParticles * sizeof(*mass));
        cudaMalloc((void **) &shellDepthFraction, numParticles * sizeof(*shellDepthFraction));
        cudaMalloc((void **) &elasticSpringConstant, numParticles * sizeof(*elasticSpringConstant));
        cudaMalloc((void **) &inelasticSpringForceReductionFactor,
                   numParticles * sizeof(*inelasticSpringForceReductionFactor));
//        cudaMalloc((void **) &_numParticles,
//                   numParticles * sizeof(*_numParticles));

        // Allocate helper struct on the device
        Particles_cuda *p_cuda;
        cudaMalloc((void **) &p_cuda, sizeof(*p_cuda));

        // Copy data from host to device
        cudaMemcpy(type, this->type, numParticles * sizeof(*type), cudaMemcpyHostToDevice);
//        cudaMemcpy(pos, this->pos, numParticles * sizeof(*pos), cudaMemcpyHostToDevice);
        cudaMemcpy(velo, this->velo, numParticles * sizeof(*velo), cudaMemcpyHostToDevice);
        cudaMemcpy(accel, this->accel, numParticles * sizeof(*accel), cudaMemcpyHostToDevice);
        cudaMemcpy(radius, this->radius, numParticles * sizeof(*radius), cudaMemcpyHostToDevice);
        cudaMemcpy(mass, this->mass, numParticles * sizeof(*mass), cudaMemcpyHostToDevice);
        cudaMemcpy(shellDepthFraction, this->shellDepthFraction, numParticles * sizeof(*shellDepthFraction),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(elasticSpringConstant, this->elasticSpringConstant, numParticles * sizeof(*elasticSpringConstant),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(inelasticSpringForceReductionFactor, this->inelasticSpringForceReductionFactor,
                   numParticles * sizeof(*inelasticSpringForceReductionFactor), cudaMemcpyHostToDevice);
//        cudaMemcpy(_numParticles, (int*) &numParticles,
//                   numParticles * sizeof(*_numParticles), cudaMemcpyHostToDevice);

        // NOTE: Binding pointers with p_cuda
        cudaMemcpy(&(p_cuda->type), &type, sizeof(p_cuda->type), cudaMemcpyHostToDevice);
//        cudaMemcpy(&(p_cuda->pos), &pos, sizeof(p_cuda->pos), cudaMemcpyHostToDevice);
        cudaMemcpy(&(p_cuda->velo), &velo, sizeof(p_cuda->velo), cudaMemcpyHostToDevice);
        cudaMemcpy(&(p_cuda->accel), &accel, sizeof(p_cuda->accel), cudaMemcpyHostToDevice);
        cudaMemcpy(&(p_cuda->radius), &radius, sizeof(p_cuda->radius), cudaMemcpyHostToDevice);
        cudaMemcpy(&(p_cuda->mass), &mass, sizeof(p_cuda->mass), cudaMemcpyHostToDevice);
        cudaMemcpy(&(p_cuda->shellDepthFraction), &shellDepthFraction, sizeof(p_cuda->shellDepthFraction),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(&(p_cuda->elasticSpringConstant), &elasticSpringConstant, sizeof(p_cuda->elasticSpringConstant),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(&(p_cuda->inelasticSpringForceReductionFactor), &inelasticSpringForceReductionFactor,
                   sizeof(p_cuda->inelasticSpringForceReductionFactor), cudaMemcpyHostToDevice);
//        cudaMemcpy(&(p_cuda->numParticles), &_numParticles,
//                   sizeof(p_cuda->numParticles), cudaMemcpyHostToDevice);
        return p_cuda;
    }


    int numParticles = 0;


    TYPE *type = nullptr;

    /// can't use a std::vector, since the gl buffer mapping to cpu space gives an allocated pointer already,
    /// which has to be used.
    //unit: km
    glm::vec4 *pos = nullptr;
    //unit: km/s
    glm::vec4 *velo = nullptr;
    //unit: km/s^2
    glm::vec4 *accel = nullptr;
    GLfloat *radius = nullptr;

    //unit: kg
    GLfloat *mass = nullptr;

    GLfloat *shellDepthFraction = nullptr; //P: SDP
    GLfloat *&SDF = shellDepthFraction;

    //unit: kg /(km*s^2) (km to be consistant with other length scales in the simulation)
    GLfloat *elasticSpringConstant = nullptr; //P: K
    GLfloat *&K = elasticSpringConstant;

    GLfloat *inelasticSpringForceReductionFactor = nullptr; //P: KRP
    GLfloat *&KRF = inelasticSpringForceReductionFactor;


    void setParticleType(int particleNum, TYPE type, float radius, float massAdjustmentFactor) { //KARL_TODO: enter corrected values
        this->radius[particleNum] = radius;
        switch (type) {
            case Particles::TYPE::SILICATE:
                this->mass[particleNum] = static_cast<GLfloat>(massAdjustmentFactor * 7.4161E19 * pow(radius / 188.39, 3));
                this->SDF[particleNum] = static_cast<GLfloat>(1 - 0.001);
                this->K[particleNum] = 2.9114E14; // TODO: Should probably scale with radius somehow
                this->KRF[particleNum] = 0.01;
                break;
            case Particles::TYPE::IRON:
                this->mass[particleNum] = static_cast<GLfloat>(massAdjustmentFactor * 1.9549E20 * pow(radius / 188.39, 3));
                this->SDF[particleNum] = static_cast<GLfloat>(1 - 0.002);
                this->K[particleNum] = 5.8228E14; // TODO: Should probably scale with radius somehow
                this->KRF[particleNum] = 0.02;
                break;
        }
    }

    int sizeInBytes() {
        return numParticles * sizeof(TYPE) +
               numParticles * sizeof(glm::vec4) +
               numParticles * sizeof(glm::vec4) +
               numParticles * sizeof(glm::vec4) +
               numParticles * sizeof(GLfloat) +
               numParticles * sizeof(GLfloat) +
               numParticles * sizeof(GLfloat) +
               numParticles * sizeof(GLfloat) +
               numParticles * sizeof(GLfloat);
    }

    void setParticlePos(glm::vec4 *particlesPos) {
        delete[](pos);
        pos = particlesPos;
    }

    void clear() {
        delete[] type;
        delete[] pos;
        delete[] velo;
        delete[] accel;
        delete[] radius;
        delete[] shellDepthFraction;
        delete[] elasticSpringConstant;
        delete[] inelasticSpringForceReductionFactor;

    }

};
