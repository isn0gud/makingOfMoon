#version 450 core

in flat uint instanceId;
out vec4 FragColor;

uniform vec4 inColor;

layout(binding = 2) buffer particle_attribute_ssbo
{
    // Stores color and radius as a vector: color = (red, green, blue, radius) 
    vec4 attrib[]; 
} attribBuff;

void main() {
    FragColor = vec4(attribBuff.attrib[instanceId].x, attribBuff.attrib[instanceId].y, attribBuff.attrib[instanceId].z, 1) - inColor;
}
