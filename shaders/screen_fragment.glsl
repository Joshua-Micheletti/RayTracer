#version 430 core
out vec4 FragColor;
	
in vec2 TexCoords;
in float frag_frames;
	
uniform sampler2D tex;
uniform sampler2D old_tex;


const float HCV_EPSILON = 1e-10;
const float HSL_EPSILON = 1e-10;
const float HCY_EPSILON = 1e-10;

const float SRGB_GAMMA = 1.0 / 2.2;
const float SRGB_INVERSE_GAMMA = 2.2;
const float SRGB_ALPHA = 0.055;


float linear_to_srgb(float channel) {
    if(channel <= 0.0031308)
        return 12.92 * channel;
    else
        return (1.0 + SRGB_ALPHA) * pow(channel, 1.0/2.4) - SRGB_ALPHA;
}

// Converts a single srgb channel to rgb
float srgb_to_linear(float channel) {
    if (channel <= 0.04045)
        return channel / 12.92;
    else
        return pow((channel + SRGB_ALPHA) / (1.0 + SRGB_ALPHA), 2.4);
}


vec3 rgb_to_srgb(vec3 rgb) {
    return vec3(
        linear_to_srgb(rgb.r),
        linear_to_srgb(rgb.g),
        linear_to_srgb(rgb.b)
    );
}


vec3 srgb_to_rgb(vec3 srgb) {
    return vec3(
        srgb_to_linear(srgb.r),
        srgb_to_linear(srgb.g),
        srgb_to_linear(srgb.b)
    );
}


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

    vec3 texCol = srgb_to_rgb(texture(tex, TexCoords).rgb);
    vec3 old_texCol = srgb_to_rgb(texture(old_tex, TexCoords).rgb);

    float weight = 1.0 / (frag_frames + 1);

    vec3 accumulated = old_texCol * (1 - weight) + texCol * weight;
    // vec3 accumulated = old_texCol + texCol;


    FragColor = vec4(rgb_to_srgb(accumulated), 1.0);
    // FragColor = vec4(vec3(weight), 1.0);
    // FragColor = vec4(average_col, 1.0);
    // FragColor = vec4(1.0);
    // FragColor = vec4(TexCoords.x, TexCoords.y, 0.0, 1.0);
}