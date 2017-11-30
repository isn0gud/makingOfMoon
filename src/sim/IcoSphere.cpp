
#include "IcoSphere.hpp"

#include <map>
#include <array>
#include "IcoSphere.hpp"



//courtesy: https://schneide.wordpress.com/2016/07/15/generating-an-icosphere-in-c/




namespace icosahedron {
    const float X = .525731112119133606f;
    const float Z = .850650808352039932f;
    const float N = 0.f;

    static const VertexList vertices =
            {
                    glm::vec3(-X, N, Z),
                    glm::vec3(X, N, Z),
                    glm::vec3(-X, N, -Z),
                    glm::vec3(X, N, -Z),
                    glm::vec3(N, Z, X),
                    glm::vec3(N, Z, -X),
                    glm::vec3(N, -Z, X),
                    glm::vec3(N, -Z, -X),
                    glm::vec3(Z, X, N),
                    glm::vec3(-Z, X, N),
                    glm::vec3(Z, -X, N),
                    glm::vec3(-Z, -X, N)
            };

    static const TriangleList triangles =
            {
                    {0,  4,  1},
                    {0,  9,  4},
                    {9,  5,  4},
                    {4,  5,  8},
                    {4,  8,  1},
                    {8,  10, 1},
                    {8,  3,  10},
                    {5,  3,  8},
                    {5,  2,  3},
                    {2,  7,  3},
                    {7,  10, 3},
                    {7,  6,  10},
                    {7,  11, 6},
                    {11, 0,  6},
                    {0,  1,  6},
                    {6,  1,  10},
                    {9,  0,  11},
                    {9,  11, 2},
                    {9,  2,  5},
                    {7,  2,  11}
            };
}

using Lookup=std::map<std::pair<Index, Index>, Index>;

Index vertex_for_edge(Lookup &lookup,
                      VertexList &vertices, Index first, Index second) {
    Lookup::key_type key(first, second);
    if (key.first > key.second)
        std::swap(key.first, key.second);

    auto inserted = lookup.insert({key, vertices.size()});
    if (inserted.second) {
        auto &edge0 = vertices[first];
        auto &edge1 = vertices[second];
        auto point = glm::normalize(edge0 + edge1);
        vertices.push_back(point);
    }

    return inserted.first->second;
}


TriangleList subdivide(VertexList &vertices,
                       TriangleList triangles) {
    Lookup lookup;
    TriangleList result;

    for (auto &&each:triangles) {
        Index mid[3];
        for (int edge = 0; edge < 3; ++edge) {
            mid[edge] = vertex_for_edge(lookup, vertices,
                                        each.vertex[edge], each.vertex[(edge + 1) % 3]);
        }

        result.push_back({each.vertex[0], mid[0], mid[2]});
        result.push_back({each.vertex[1], mid[1], mid[0]});
        result.push_back({each.vertex[2], mid[2], mid[1]});
        result.push_back({mid[0], mid[1], mid[2]});
    }

    return result;
}


IndexedMesh make_icosphere(int subdivisions) {
    VertexList vertices = icosahedron::vertices;
    TriangleList triangles = icosahedron::triangles;

    for (int i = 0; i < subdivisions; ++i) {
        triangles = subdivide(vertices, triangles);
    }

    return {vertices, triangles};
}

IndexedMesh getIcoSphere(glm::vec3 center, float radius, int num) {
    //TODO create function with radius and number
    //Creates x number of particles determined by the given division.
    // All Particles are uniformly spread on a sphere with radius 1.
    // So, to change the radius the vertices have to be with that radius (pos_vecotr * 2 = radius of 2).
    // I was thinking to do layers.
    //1 = 24
    //2 = 162
    //3 = 642
    //4 = 2562
    return make_icosphere(1);
}
