#version 450 core

layout (location = 0) in vec3 aPos;

buffer particles_ssbo
{
    vec4 pos[];
};

uniform mat4 mvp;
uniform float radius;

void main() {
    gl_Position = mvp * (pos[gl_InstanceID] + vec4(radius*aPos, 1));
}
