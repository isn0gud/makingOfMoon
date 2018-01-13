//#pragma once
//
//
//#include <cuda_runtime_api.h>
//#include "common.hpp"
//#include "Constants.h"
//
//
//class Particles {
//
//public:
//
//    int numParticles = 0;
//
//    TYPE *type;
//
//    /// can't use a std::vector, since the gl buffer mapping to cpu space gives an allocated pointer already,
//    /// which has to be used.
//    //unit: km
//    glm::vec4 *pos;
//    //unit: km/s
//    glm::vec4 *velo;
//    //unit: km/s^2
//    glm::vec4 *accel;
//    ParticleConst pConst;
//
//public:
//
//
//    explicit Particles(int numParticles) : numParticles(numParticles) {
//        type = new TYPE[numParticles];
//        pos = new glm::vec4[numParticles];
//        velo = new glm::vec4[numParticles];
//        accel = new glm::vec4[numParticles];
//
//    }
//
//    typedef struct Particles_cuda {
//        TYPE *type;
////        glm::vec4 *pos;
//        glm::vec4 *velo;
//        glm::vec4 *accel;
//
//    } Particles_cuda;
//
//    Particles_cuda *to_cuda() {
//        TYPE *type;
////        glm::vec4 *pos, *velo, *accel;
//        glm::vec4 *velo, *accel;
//
//        // Allocate device data
//        cudaMalloc((void **) &type, numParticles * sizeof(*type));
////        cudaMalloc((void **) &pos, numParticles * sizeof(*pos));
//        cudaMalloc((void **) &velo, numParticles * sizeof(*velo));
//        cudaMalloc((void **) &accel, numParticles * sizeof(*accel));
//
//        // Allocate helper struct on the device
//        Particles_cuda *p_cuda;
//        cudaMalloc((void **) &p_cuda, sizeof(*p_cuda));
//
//        // Copy data from host to device
//        cudaMemcpy(type, this->type, numParticles * sizeof(*type), cudaMemcpyHostToDevice);
////        cudaMemcpy(pos, this->pos, numParticles * sizeof(*pos), cudaMemcpyHostToDevice);
//        cudaMemcpy(velo, this->velo, numParticles * sizeof(*velo), cudaMemcpyHostToDevice);
//        cudaMemcpy(accel, this->accel, numParticles * sizeof(*accel), cudaMemcpyHostToDevice);
//
//        // NOTE: Binding pointers with p_cuda
//        cudaMemcpy(&(p_cuda->type), &type, sizeof(p_cuda->type), cudaMemcpyHostToDevice);
////        cudaMemcpy(&(p_cuda->pos), &pos, sizeof(p_cuda->pos), cudaMemcpyHostToDevice);
//        cudaMemcpy(&(p_cuda->velo), &velo, sizeof(p_cuda->velo), cudaMemcpyHostToDevice);
//        cudaMemcpy(&(p_cuda->accel), &accel, sizeof(p_cuda->accel), cudaMemcpyHostToDevice);
//        return p_cuda;
//    }
//
//    void setParticleType(int particleNum, TYPE type) {
//        this->type[particleNum] = type;
//    }
//
//    void setParticlePos(glm::vec4 *particlesPos) {
//        delete[](pos);
//        pos = particlesPos;
//    }
//
//    void clear() {
//        delete[] type;
//        //delete[] pos;
//        delete[] velo;
//        delete[] accel;
//    }
//
//};
