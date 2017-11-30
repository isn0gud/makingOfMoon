//
// Created by Pius Friesch on 29/11/2017.
//

#ifndef ASS_OPENGL_ICOSPHERE_HPP
#define ASS_OPENGL_ICOSPHERE_HPP

#include "../common.hpp"

using Index=int;

struct Triangle {
    Index vertex[3];
};

using TriangleList=std::vector<Triangle>;
using VertexList=std::vector<glm::vec3>;

using IndexedMesh=std::pair<VertexList, TriangleList>;


IndexedMesh getIcoSphere(glm::vec3 center, float radius, int num);


#endif //ASS_OPENGL_ICOSPHERE_HPP
