#version 330 core

in vec2 position;
in float hit;
in vec3 debug;

out vec4 fragColor;

float map(float x, float in_min, float in_max, float out_min, float out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

void main() {
    /*
    if (hit == 1.0) {
        fragColor = vec4(map(position.x, -1, 1, 0, 1), map(position.y, -1, 1, 0, 1), 0.0f, 1.0f);
    } else {
        fragColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }*/

    /*
    if (position.x < 0.6 && position.x > 0.4) {
        fragColor = vec4(map(position.x, -1, 1, 0, 1), map(position.y, -1, 1, 0, 1), 0.0f, 1.0f);
    } else {
        fragColor = vec4(0, 0, 0, 1);
    }*/

    //fragColor = vec4(hit, hit, hit, 1.0);
    
    fragColor = vec4(debug.x, debug.y, debug.z, 1.0);
}




