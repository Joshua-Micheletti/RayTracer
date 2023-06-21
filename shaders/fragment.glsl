#version 460 core

in vec2 position;
in mat4 fragModel;
in mat4 view;

out vec4 fragColor;

layout (std430, binding = 1) buffer shader_data
{ 
    float[] vertices;
};


float map(float x, float in_min, float in_max, float out_min, float out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

float inside_outside_test(vec3 v0, vec3 v1, vec3 v2, vec3 p, vec3 n) {
    vec3 edge0 = v1 - v0;
    vec3 edge1 = v2 - v1;
    vec3 edge2 = v0 - v2;

    vec3 c0 = p - v0;
    vec3 c1 = p - v1;
    vec3 c2 = p - v2;

    if (dot(n, cross(edge0, c0)) > 0.0 &&
        dot(n, cross(edge1, c1)) > 0.0 &&
        dot(n, cross(edge2, c2)) > 0.0) {
            return(1.0);
    } else {
        return(-1.0);
    }
}


void main() {

    float hit = 0;

    int i;

    for (i = 0; i <= vertices.length(); i += 9) {
        vec4 v0_0 = vec4(vertices[i    ], vertices[i + 1], vertices[i + 2], 1.0);
        vec4 v1_0 = vec4(vertices[i + 3], vertices[i + 4], vertices[i + 5], 1.0);
        vec4 v2_0 = vec4(vertices[i + 6], vertices[i + 7], vertices[i + 8], 1.0);

        v0_0 = view * fragModel * v0_0;
        v1_0 = view * fragModel * v1_0;
        v2_0 = view * fragModel * v2_0;

        vec3 v0 = v0_0.xyz;
        vec3 v1 = v1_0.xyz;
        vec3 v2 = v2_0.xyz;
        

        // v0 *= fragModel;
        // v1 *= fragModel;
        // v2 *= fragModel;

        vec3 edge01 = v1 - v0;
        vec3 edge02 = v2 - v0;

        vec3 normal = normalize(cross(edge01, edge02));

        // vec3 camera = vec3(0.0, 0.0, -3.0);
        vec3 camera = vec3(0.0, 0.0, -3.0);

        vec3 pixel = vec3(position.x, position.y, -2.0);

        vec3 origin = camera;
        vec3 direction = normalize(pixel - camera);

        float dist = -dot(normal, v0);

        float parallelism = dot(normal, direction);

        if (parallelism != 0.0) {
            float t = -(dot(normal, origin) + dist) / parallelism;

            if (t > 0.0) {
                vec3 p_hit = origin + (t * direction);

                if (inside_outside_test(v0, v1, v2, p_hit, normal) == 1.0) {
                    hit += 1.0;
                }
            }
        }
    }

    // fragColor = vec4(hit, hit, hit, 1.0);
    if (hit > 0) {
        fragColor = vec4(0.2, 0.5, 1.0, 1.0);
        // fragColor = vec4(map(hit, 0, 3, 0, 1), 0.0, 0.0, 1.0);
    } else {
        fragColor = vec4(0.0);
    }

    // fragColor = vec4(vertices.length(), 0.0, 0.0, 0.0);
    // fragColor = vec4(map(float(i), 0, 18, 0, 1), 0.0, 0.0, 1.0);
}




