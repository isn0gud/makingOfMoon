#version 450 core

in vec3 aPos;
out flat uint instanceId;

layout(binding = 1) buffer particle_position_ssbo
{
    vec4 pos[];
} posBuff;


layout(binding = 2) buffer particle_attribute_ssbo
{
    // Stores color and radius as a vector: color = (red, green, blue, radius) 
    vec4 attrib[]; 
} attribBuff;

uniform mat4 mvp;

void main() {
    instanceId = gl_InstanceID;
    gl_Position = mvp * (posBuff.pos[gl_InstanceID] + vec4(attribBuff.attrib[gl_InstanceID].w * aPos, 1));
}
