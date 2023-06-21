#version 460 core

layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform vec3 camera;


out vec2 position;
out mat4 fragModel;
out vec3 fragCamera;

void main() {
    position = vec2(aPos.xy);
    fragModel = model;
    fragCamera = camera;
    gl_Position = vec4(aPos, 1.0);
}