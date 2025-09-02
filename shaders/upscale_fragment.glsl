#version 430 core
out vec4 FragColor;
	
in vec2 TexCoords;

uniform sampler2D tex;

void main() {
    float sharpness = 0.0;

    vec2 tex_offset = 1.0 / textureSize(tex, 0); // pixel size
    vec3 color = texture(tex, TexCoords).rgb * (1.0 + 4.0 * sharpness);
    color -= texture(tex, TexCoords + vec2(tex_offset.x, 0.0)).rgb * sharpness;
    color -= texture(tex, TexCoords - vec2(tex_offset.x, 0.0)).rgb * sharpness;
    color -= texture(tex, TexCoords + vec2(0.0, tex_offset.y)).rgb * sharpness;
    color -= texture(tex, TexCoords - vec2(0.0, tex_offset.y)).rgb * sharpness;
    FragColor = vec4(color, 1.0);
}
