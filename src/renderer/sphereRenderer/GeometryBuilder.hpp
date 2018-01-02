#pragma once

#include "../../common.hpp"
#include <vector>




class GeometryBuilder {
private:
    GeometryBuilder() {}

    ~GeometryBuilder() {}

public:
    static void buildSphere(int nSections, float radius, vector<vec3> &outVertices);

    static void buildCube(float side, vector<vec3> &outVertices);
};

