//#pragma once
//
//#include "common.hpp"
//
//
//#define MASS_SCALING 1.0f
//#define DIST_SCALING 1.0f
//#define TYPE_NUM 2
//
////  units: SI, but km instead of m
////  6.674×10−20 (km)^3⋅kg^(−1)⋅s^(−2)
//#define GRAV_CONST (6.674E20f * (DIST_SCALING * DIST_SCALING * DIST_SCALING) / (MASS_SCALING * MASS_SCALING))
//#define DIST_EPSILON (47.0975f / DIST_SCALING)
//
////used for array index, don't change vals
//enum TYPE {
//    IRON = 0,
//    SILICATE = 1
//};
//
//
//struct ParticleConst {
//    float mass;
//    float radius;
//
//    float shellDepthFraction;
//
//    float elasticSpringConstant;
//
//    float inelasticSpringForceReductionFactor;
//};
//
////class ParticleConstVals {
////
////
////public:
////    static
//static ParticleConst pConsts[NUM_PLANETS * TYPE_NUM];
//
//static void  setParticleConstants(int planetNum,
//                          float radius,
//                          float massAdjustmentFactor) {
//    pConsts[planetNum * TYPE::IRON] = {
//
//            .mass= massAdjustmentFactor * (1.9549E20f / MASS_SCALING) *
//                   static_cast<GLfloat>(pow(radius / (188.39f / DIST_SCALING), 3.0f)),
//            .radius = radius,
//            .shellDepthFraction=  1 - 0.002f,
//            .elasticSpringConstant=5.8228E14f / MASS_SCALING,
//            .inelasticSpringForceReductionFactor=0.02f};
//
//    pConsts[planetNum * TYPE::SILICATE] = {
//            mass: massAdjustmentFactor * (7.4161E19f / MASS_SCALING) *
//                  static_cast<GLfloat>(pow(radius / (188.39 / DIST_SCALING), 3)),
//            .radius = radius,
//
//            shellDepthFraction: 1 - 0.001f,
//            elasticSpringConstant:2.9114E14f / MASS_SCALING,
//            inelasticSpringForceReductionFactor:0.01};
//
//
//};
//
//
////};
