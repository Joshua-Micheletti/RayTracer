#version 430 core
out vec4 FragColor;
	
in vec2 TexCoords;
in float frag_frames;
	
uniform sampler2D tex;
uniform sampler2D old_tex;
	
void main()
{
    // vec3 average_col = vec3(0, 0, 0);

    // float pixel_x = 1.0 / 384.0;
    // float pixel_y = 1.0 / 216.0;

    // average_col += texture(tex, vec2(TexCoords.x - pixel_x, TexCoords.y - pixel_y)).rgb;
    // average_col += texture(tex, vec2(TexCoords.x - pixel_x, TexCoords.y    )).rgb;
    // average_col += texture(tex, vec2(TexCoords.x - pixel_x, TexCoords.y + pixel_y)).rgb;
    // average_col += texture(tex, vec2(TexCoords.x    , TexCoords.y - pixel_y)).rgb;
    // average_col += texture(tex, vec2(TexCoords.x    , TexCoords.y    )).rgb;
    // average_col += texture(tex, vec2(TexCoords.x    , TexCoords.y + pixel_y)).rgb;
    // average_col += texture(tex, vec2(TexCoords.x + pixel_x, TexCoords.y - pixel_y)).rgb;
    // average_col += texture(tex, vec2(TexCoords.x + pixel_x, TexCoords.y    )).rgb;
    // average_col += texture(tex, vec2(TexCoords.x + pixel_x, TexCoords.y + pixel_y)).rgb;

    // average_col += texture(tex, vec2(TexCoords.x - pixel_x, TexCoords.y)).rgb;
    // average_col += texture(tex, vec2(TexCoords.x + pixel_x, TexCoords.y)).rgb;
    // average_col += texture(tex, vec2(TexCoords.x, TexCoords.y - pixel_y)).rgb;
    // average_col += texture(tex, vec2(TexCoords.x, TexCoords.y + pixel_y)).rgb;

    // average_col = average_col / 4;

    vec3 texCol = texture(tex, TexCoords).rgb;
    vec3 old_texCol = texture(old_tex, TexCoords).rgb;

    float weight = 1.0 / (frag_frames + 1);

    vec3 accumulated = old_texCol * (1 - weight) + texCol * weight;
    // vec3 accumulated = old_texCol + texCol;


    FragColor = vec4(accumulated, 1.0);
    // FragColor = vec4(vec3(weight), 1.0);
    // FragColor = vec4(average_col, 1.0);
    // FragColor = vec4(1.0);
    // FragColor = vec4(TexCoords.x, TexCoords.y, 0.0, 1.0);
}