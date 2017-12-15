#include "GeometryBuilder.hpp"
#include <math.h>
#include <iostream>

#define PI 3.14159265359

using namespace std;


void GeometryBuilder::buildSphere(int nSections, float radius, vector<vec3>& outVertices)
{
    outVertices.resize(3*nSections*nSections);
    for(int i = 0; i < nSections/2; i++)
    {
        for(int j = 0; j < nSections; j++)
        {
            int index = 6*(i*nSections + j);
            float polarAngle1 = PI*(((float)i)/((float)nSections/2));
            float azimuthAngle1 = 2*PI*((float)j)/((float)nSections);
            float polarAngle2 = PI*(((float)i+1)/((float)nSections/2));
            float azimuthAngle2 = 2*PI*((float)j+1)/((float)nSections);

            outVertices[index].x = radius*((float)(cos(azimuthAngle1)*sin(polarAngle1)));
            outVertices[index].y = radius*((float)(sin(azimuthAngle1)*sin(polarAngle1)));
            outVertices[index].z = radius*((float)(cos(polarAngle1)));

            outVertices[index+1].x = radius*((float)(cos(azimuthAngle1)*sin(polarAngle2)));
            outVertices[index+1].y = radius*((float)(sin(azimuthAngle1)*sin(polarAngle2)));
            outVertices[index+1].z = radius*((float)(cos(polarAngle2)));

            outVertices[index+2].x = radius*((float)(cos(azimuthAngle2)*sin(polarAngle2)));
            outVertices[index+2].y = radius*((float)(sin(azimuthAngle2)*sin(polarAngle2)));
            outVertices[index+2].z = radius*((float)(cos(polarAngle2)));

            outVertices[index+3].x = radius*((float)(cos(azimuthAngle1)*sin(polarAngle1)));
            outVertices[index+3].y = radius*((float)(sin(azimuthAngle1)*sin(polarAngle1)));
            outVertices[index+3].z = radius*((float)(cos(polarAngle1)));

            outVertices[index+4].x = radius*((float)(cos(azimuthAngle2)*sin(polarAngle2)));
            outVertices[index+4].y = radius*((float)(sin(azimuthAngle2)*sin(polarAngle2)));
            outVertices[index+4].z = radius*((float)(cos(polarAngle2)));

            outVertices[index+5].x = radius*((float)(cos(azimuthAngle2)*sin(polarAngle1)));
            outVertices[index+5].y = radius*((float)(sin(azimuthAngle2)*sin(polarAngle1)));
            outVertices[index+5].z = radius*((float)(cos(polarAngle1)));
        }
    }
}


void GeometryBuilder::buildCube(float side, vector<vec3> &outVertices)
{
    outVertices.resize(36);
    outVertices[0] = vec3(-side/2, side/2, side/2);
    outVertices[1] = vec3(-side/2, -side/2, side/2);
    outVertices[2] = vec3(side/2, side/2, side/2);
    outVertices[3] = vec3(side/2, side/2, side/2);
    outVertices[4] = vec3(-side/2, -side/2, side/2);
    outVertices[5] = vec3(side/2, -side/2, side/2);

    outVertices[6] = vec3(side/2, side/2, side/2);
    outVertices[7] = vec3(side/2, -side/2, side/2);
    outVertices[8] = vec3(side/2, -side/2, -side/2);
    outVertices[9] = vec3(side/2, side/2, side/2);
    outVertices[10] = vec3(side/2, -side/2, -side/2);
    outVertices[11] = vec3(side/2, side/2, -side/2);

    outVertices[12] = vec3(side/2, side/2, -side/2);
    outVertices[13] = vec3(side/2, -side/2, -side/2);
    outVertices[14] = vec3(-side/2, -side/2, -side/2);
    outVertices[15] = vec3(side/2, side/2, -side/2);
    outVertices[16] = vec3(-side/2, -side/2, -side/2);
    outVertices[17] = vec3(-side/2, side/2, -side/2);

    outVertices[18] = vec3(-side/2, side/2, -side/2);
    outVertices[19] = vec3(-side/2, -side/2, -side/2);
    outVertices[20] = vec3(-side/2, -side/2, side/2);
    outVertices[21] = vec3(-side/2, side/2, -side/2);
    outVertices[22] = vec3(-side/2, -side/2, side/2);
    outVertices[23] = vec3(-side/2, side/2, side/2);

    outVertices[24] = vec3(-side/2, side/2, side/2);
    outVertices[25] = vec3(side/2, side/2, side/2);
    outVertices[26] = vec3(side/2, side/2, -side/2);
    outVertices[27] = vec3(-side/2, side/2, side/2);
    outVertices[28] = vec3(side/2, side/2, -side/2);
    outVertices[29] = vec3(-side/2, side/2, -side/2);

    outVertices[30] = vec3(-side/2, -side/2, side/2);
    outVertices[31] = vec3(side/2, -side/2, -side/2);
    outVertices[32] = vec3(side/2, -side/2, side/2);
    outVertices[33] = vec3(-side/2, -side/2, side/2);
    outVertices[34] = vec3(-side/2, -side/2, -side/2);
    outVertices[35] = vec3(side/2, -side/2, -side/2);
}
