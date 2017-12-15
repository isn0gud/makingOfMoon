#ifndef AGP_PROJECT_GEOMETRYBUILDER_HPP
#define AGP_PROJECT_GEOMETRYBUILDER_HPP

#include "../common.hpp"
#include <vector>

using namespace std;
using namespace glm;

class GeometryBuilder
{
private:
    GeometryBuilder() {}
    ~GeometryBuilder() {}
public:
    static void buildSphere(int nSections, float radius, vector<vec3>& outVertices);
    static void buildCube(float side, vector<vec3>& outVertices);
};

#endif
