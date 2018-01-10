#version 450 core

layout (location = 0) in vec3 aPos;

buffer particles_ssbo
{
    vec4 pos[];
};

uniform mat4 mvp;

void main(){
    float radius = 84.2497;
    /*gl_Position = mvp * (vec4(radius*aPos, 1));*/
    gl_Position = mvp * (pos[gl_InstanceID] + vec4(radius*aPos, 1));
}
