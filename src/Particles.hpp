#pragma oncecumulative<cuda_runtime_api.h>
#include "common.hpp"

#define IRON_shellDepthFraction (1 - 0.002f)
#define IRON_elasticSpringConstant 5.8228E14
#define IRON_inelasticSpringForceReductionFactor 0.02f

#define SILICATE_shellDepthFraction (1 - 0.001f)
#define SILICATE_elasticSpringConstant 2.9114E14
#define SILICATE_inelasticSpringForceReductionFactor  0.01


//  units: SI, but km instead of m
//  6.674×10−20 (km)^3⋅kg^(−1)⋅s^(−2)
#define G 6.674E-20
#define distanceEpsilon 47.0975


enum TYPE {
    IRON,
    SILICATE
};

class Particles {
public:
    class ParticlesInit {
    public:
        int numParticles = 0;
        std::vector<TYPE> type;
        std::vector<glm::vec4> pos__radius;
        std::vector<glm::vec4> velo__mass;

        void setParticleType(int particleNum, TYPE type, float radius,
                             float massAdjustmentFactor) {
            this->pos__radius[particleNum].w = radius;
            this->type.push_back(type);
            switch (type) {
                case TYPE::SILICATE:
                    this->velo__mass[particleNum].w = static_cast<GLfloat>(massAdjustmentFactor * (7.4161E19) *
                                                                           pow(radius / (188.39), 3));
                    break;
                case TYPE::IRON:
                    this->velo__mass[particleNum].w = static_cast<GLfloat>(massAdjustmentFactor * (1.9549E20) *
                                                                           pow(radius / (188.39), 3));
                    break;
            }
        }

        void addParticles(ParticlesInit &other) {
            numParticles += other.numParticles;
            type.insert(type.end(), other.type.begin(), other.type.end());
            pos__radius.insert(pos__radius.end(), other.pos__radius.begin(), other.pos__radius.end());
            velo__mass.insert(velo__mass.end(), other.velo__mass.begin(), other.velo__mass.end());

        }

        void clear() {
            numParticles = 0;
            type.clear();
            pos__radius.clear();
            velo__mass.clear();
        }
    };


public:
    explicit Particles(ParticlesInit initData) {
        numParticles = initData.numParticles;
        type = new TYPE[numParticles];
        memcpy(type, initData.type.data(), numParticles * sizeof(TYPE));

        pos__radius = new glm::vec4[numParticles];
        memcpy(pos__radius, initData.pos__radius.data(), numParticles * sizeof(glm::vec4));

        velo__mass = new glm::vec4[numParticles];
        memcpy(velo__mass, initData.velo__mass.data(), numParticles * sizeof(glm::vec4));

    }

    static glm::vec3 getMaterialColor(TYPE materialType) {
        if (materialType == IRON)
            return glm::vec3(0.85, 0.25, 0.25);
        if (materialType == SILICATE)
            return glm::vec3(0.75, 0.65, 0.35);
        return glm::vec3(0, 0, 0);
    }

    typedef struct Particles_cuda {
        TYPE *type;
        glm::vec4 *velo__mass;

        int *numParticles;
    } Particles_cuda;

    Particles_cuda *to_cuda() {
        TYPE *type;
        glm::vec4 *velo;
        int *_numParticles;

        // Allocate device data
        cudaMalloc((void **) &_numParticles, sizeof(*_numParticles));
        cudaMalloc((void **) &type, numParticles * sizeof(*type));
        cudaMalloc((void **) &velo, numParticles * sizeof(*velo));

        // Allocate helper struct on the device
        Particles_cuda *p_cuda;
        cudaMalloc((void **) &p_cuda, sizeof(*p_cuda));

        // Copy data from host to device
        cudaMemcpy(_numParticles, &numParticles, sizeof(*_numParticles), cudaMemcpyHostToDevice);
        cudaMemcpy(type, this->type, numParticles * sizeof(*type), cudaMemcpyHostToDevice);
        cudaMemcpy(velo, this->velo__mass, numParticles * sizeof(*velo), cudaMemcpyHostToDevice);

        //Binding pointers with p_cuda
        cudaMemcpy(&(p_cuda->numParticles), &_numParticles, sizeof(p_cuda->numParticles), cudaMemcpyHostToDevice);
        cudaMemcpy(&(p_cuda->type), &type, sizeof(p_cuda->type), cudaMemcpyHostToDevice);
        cudaMemcpy(&(p_cuda->velo__mass), &velo, sizeof(p_cuda->velo__mass), cudaMemcpyHostToDevice);

        return p_cuda;
    }

    int numParticles = 0;

    TYPE *type;
    glm::vec4 *pos__radius;
    glm::vec4 *velo__mass;

//    int sizeInBytes() {
//        return numParticles * sizeof(TYPE) +
//               numParticles * sizeof(glm::vec4) +
//               numParticles * sizeof(glm::vec4) +
//               numParticles * sizeof(glm::vec4);
//    }

    void setParticlePos(glm::vec4 *particlesPos) {
        delete[](pos__radius);
        pos__radius = particlesPos;
    }

    void clear() {
        delete[] type;
        delete[] velo__mass;
    }

};

