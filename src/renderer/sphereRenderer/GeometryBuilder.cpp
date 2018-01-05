#include "GeometryBuilder.hpp"
#include <math.h>
#include <iostream>

#define PI 3.14159265359

void GeometryBuilder::buildSphere(int nSections, float radius, std::vector<glm::vec3> &outVertices) {
    outVertices.resize(3 * nSections * nSections);
    for (int i = 0; i < nSections / 2; i++) {
        for (int j = 0; j < nSections; j++) {
            int index = 6 * (i * nSections + j);
            float polarAngle1 = PI * (((float) i) / ((float) nSections / 2));
            float azimuthAngle1 = 2 * PI * ((float) j) / ((float) nSections);
            float polarAngle2 = PI * (((float) i + 1) / ((float) nSections / 2));
            float azimuthAngle2 = 2 * PI * ((float) j + 1) / ((float) nSections);

            outVertices[index].x = radius * ((float) (cos(azimuthAngle1) * sin(polarAngle1)));
            outVertices[index].y = radius * ((float) (sin(azimuthAngle1) * sin(polarAngle1)));
            outVertices[index].z = radius * ((float) (cos(polarAngle1)));

            outVertices[index + 1].x = radius * ((float) (cos(azimuthAngle1) * sin(polarAngle2)));
            outVertices[index + 1].y = radius * ((float) (sin(azimuthAngle1) * sin(polarAngle2)));
            outVertices[index + 1].z = radius * ((float) (cos(polarAngle2)));

            outVertices[index + 2].x = radius * ((float) (cos(azimuthAngle2) * sin(polarAngle2)));
            outVertices[index + 2].y = radius * ((float) (sin(azimuthAngle2) * sin(polarAngle2)));
            outVertices[index + 2].z = radius * ((float) (cos(polarAngle2)));

            outVertices[index + 3].x = radius * ((float) (cos(azimuthAngle1) * sin(polarAngle1)));
            outVertices[index + 3].y = radius * ((float) (sin(azimuthAngle1) * sin(polarAngle1)));
            outVertices[index + 3].z = radius * ((float) (cos(polarAngle1)));

            outVertices[index + 4].x = radius * ((float) (cos(azimuthAngle2) * sin(polarAngle2)));
            outVertices[index + 4].y = radius * ((float) (sin(azimuthAngle2) * sin(polarAngle2)));
            outVertices[index + 4].z = radius * ((float) (cos(polarAngle2)));

            outVertices[index + 5].x = radius * ((float) (cos(azimuthAngle2) * sin(polarAngle1)));
            outVertices[index + 5].y = radius * ((float) (sin(azimuthAngle2) * sin(polarAngle1)));
            outVertices[index + 5].z = radius * ((float) (cos(polarAngle1)));
        }
    }
}


void GeometryBuilder::buildCube(float side, std::vector<glm::vec3> &outVertices) {
    outVertices.resize(36);
    outVertices[0] = glm::vec3(-side / 2, side / 2, side / 2);
    outVertices[1] = glm::vec3(-side / 2, -side / 2, side / 2);
    outVertices[2] = glm::vec3(side / 2, side / 2, side / 2);
    outVertices[3] = glm::vec3(side / 2, side / 2, side / 2);
    outVertices[4] = glm::vec3(-side / 2, -side / 2, side / 2);
    outVertices[5] = glm::vec3(side / 2, -side / 2, side / 2);

    outVertices[6] = glm::vec3(side / 2, side / 2, side / 2);
    outVertices[7] = glm::vec3(side / 2, -side / 2, side / 2);
    outVertices[8] = glm::vec3(side / 2, -side / 2, -side / 2);
    outVertices[9] = glm::vec3(side / 2, side / 2, side / 2);
    outVertices[10] = glm::vec3(side / 2, -side / 2, -side / 2);
    outVertices[11] = glm::vec3(side / 2, side / 2, -side / 2);

    outVertices[12] = glm::vec3(side / 2, side / 2, -side / 2);
    outVertices[13] = glm::vec3(side / 2, -side / 2, -side / 2);
    outVertices[14] = glm::vec3(-side / 2, -side / 2, -side / 2);
    outVertices[15] = glm::vec3(side / 2, side / 2, -side / 2);
    outVertices[16] = glm::vec3(-side / 2, -side / 2, -side / 2);
    outVertices[17] = glm::vec3(-side / 2, side / 2, -side / 2);

    outVertices[18] = glm::vec3(-side / 2, side / 2, -side / 2);
    outVertices[19] = glm::vec3(-side / 2, -side / 2, -side / 2);
    outVertices[20] = glm::vec3(-side / 2, -side / 2, side / 2);
    outVertices[21] = glm::vec3(-side / 2, side / 2, -side / 2);
    outVertices[22] = glm::vec3(-side / 2, -side / 2, side / 2);
    outVertices[23] = glm::vec3(-side / 2, side / 2, side / 2);

    outVertices[24] = glm::vec3(-side / 2, side / 2, side / 2);
    outVertices[25] = glm::vec3(side / 2, side / 2, side / 2);
    outVertices[26] = glm::vec3(side / 2, side / 2, -side / 2);
    outVertices[27] = glm::vec3(-side / 2, side / 2, side / 2);
    outVertices[28] = glm::vec3(side / 2, side / 2, -side / 2);
    outVertices[29] = glm::vec3(-side / 2, side / 2, -side / 2);

    outVertices[30] = glm::vec3(-side / 2, -side / 2, side / 2);
    outVertices[31] = glm::vec3(side / 2, -side / 2, -side / 2);
    outVertices[32] = glm::vec3(side / 2, -side / 2, side / 2);
    outVertices[33] = glm::vec3(-side / 2, -side / 2, side / 2);
    outVertices[34] = glm::vec3(-side / 2, -side / 2, -side / 2);
    outVertices[35] = glm::vec3(side / 2, -side / 2, -side / 2);
}
