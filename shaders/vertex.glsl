#version 430 core

layout (location = 0) in vec3 aPos;

uniform mat4 inInverseViewProjection;
uniform vec3 inEye;
uniform float inLightIndex;
uniform mat4 inLightModel;

out vec2 position;
out mat4 inverseViewProjection;
out vec3 eye;
out float lightIndex;
out mat4 lightModel;

void main() {
    position = vec2(aPos.xy);
    inverseViewProjection = inInverseViewProjection;
    eye = inEye;
    lightIndex = inLightIndex;
    lightModel = inLightModel;
    gl_Position = vec4(aPos, 1.0);
}