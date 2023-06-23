#version 460 core

layout (location = 0) in vec3 aPos;

uniform mat4 inModel;
uniform mat4 inInverseViewProjection;
uniform vec3 inEye;
uniform vec3 inLight;

out vec2 position;
out mat4 model;
out mat4 inverseViewProjection;
out vec3 eye;
out vec3 light;

void main() {
    position = vec2(aPos.xy);
    model = inModel;
    inverseViewProjection = inInverseViewProjection;
    eye = inEye;
    light = inLight;
    gl_Position = vec4(aPos, 1.0);
}