#version 460 core

layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 camera;


out vec2 position;
out mat4 fragModel;
out mat4 view;

void main() {
    position = vec2(aPos.xy);
    fragModel = model;
    view = camera;
    gl_Position = vec4(aPos, 1.0);
}