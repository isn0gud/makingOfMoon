//#pragma
//
//
//#include <vector_types.h>
//
//template <typename T> struct vec3
//{
//    typedef float   Type;
//}; // dummy
//template <>           struct vec3<float>
//{
//    typedef float3  Type;
//};
//template <>           struct vec3<double>
//{
//    typedef double3 Type;
//};
//
//template <typename T> struct vec4
//{
//    typedef float   Type;
//}; // dummy
//template <>           struct vec4<float>
//{
//    typedef float4  Type;
//};
//template <>           struct vec4<double>
//{
//    typedef double4 Type;
//};
//
//
//
//inline float3
//scalevec(float3 &vector, float scalar)
//{
//    float3 rt = vector;
//    rt.x *= scalar;
//    rt.y *= scalar;
//    rt.z *= scalar;
//    return rt;
//}
//
//inline float
//normalize(float3 &vector)
//{
//    float dist = sqrtf(vector.x*vector.x + vector.y*vector.y + vector.z*vector.z);
//
//    if (dist > 1e-6)
//    {
//        vector.x /= dist;
//        vector.y /= dist;
//        vector.z /= dist;
//    }
//
//    return dist;
//}
//
//inline float
//dot(float3 v0, float3 v1)
//{
//    return v0.x*v1.x+v0.y*v1.y+v0.z*v1.z;
//}
//
//inline float3
//cross(float3 v0, float3 v1)
//{
//    float3 rt;
//    rt.x = v0.y*v1.z-v0.z*v1.y;
//    rt.y = v0.z*v1.x-v0.x*v1.z;
//    rt.z = v0.x*v1.y-v0.y*v1.x;
//    return rt;
//}