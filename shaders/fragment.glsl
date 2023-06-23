#version 460 core

in vec2 position;
in mat4 model;
in mat4 inverseViewProjection;
in vec3 eye;
in vec3 light;

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

float[5] ray_triangle_collision(vec3 ray_origin, vec3 ray_direction, vec3 v0, vec3 v1, vec3 v2) {
    vec3 p_hit = vec3(0, 0, 0);

    float hit = 0;
    
    vec3 edge01 = v1 - v0;
    vec3 edge02 = v2 - v0;

    vec3 normal = normalize(cross(edge01, edge02));

    float dist = -dot(normal, v0);

    float parallelism = dot(normal, ray_direction);

    float t = 99999999;

    if (parallelism != 0.0) {
        t = -(dot(normal, ray_origin) + dist) / parallelism;

        if (t > 0.0) {
            p_hit = ray_origin + (t * ray_direction);

            if (inside_outside_test(v0, v1, v2, p_hit, normal) == 1.0) {
                hit = 1.0;
            }
        }
    }

    float[5] result;
    result[0] = p_hit.x;
    result[1] = p_hit.y;
    result[2] = p_hit.z;
    result[3] = hit;
    result[4] = t;

    return(result);
}




void main() {

    vec3 color = vec3(0.0, 0.0, 0.0);

    // coordinate of end ray in screen space
    vec4 screenSpaceFar = vec4(position.xy, 1.0, 1.0);
    // coordinate of origin ray in screen space
    vec4 screenSpaceNear = vec4(position.xy, 0.0, 1.0);
    
    // coordinate of end ray in world space
    vec4 far = inverseViewProjection * screenSpaceFar;
    far /= far.w;
    // coordinate of origin ray in world space
    vec4 near = inverseViewProjection * screenSpaceNear;
    near /= near.w;
    
    // now we construct ray
    vec3 origin = eye;
    vec3 direction = normalize(far.xyz - near.xyz);

    float[5] nearest_hit;
    nearest_hit[4] = 9999999;

    for (int i = 0; i <= vertices.length(); i += 9) {
        vec4 v0 = vec4(vertices[i    ], vertices[i + 1], vertices[i + 2], 1.0);
        vec4 v1 = vec4(vertices[i + 3], vertices[i + 4], vertices[i + 5], 1.0);
        vec4 v2 = vec4(vertices[i + 6], vertices[i + 7], vertices[i + 8], 1.0);

        v0 = model * v0;
        v1 = model * v1;
        v2 = model * v2;

        // vec4 p_hit = ray_triangle_collision(origin, direction, v0.xyz, v1.xyz, v2.xyz);
        float[5] result = ray_triangle_collision(origin, direction, v0.xyz, v1.xyz, v2.xyz);

        if (result[3] == 1) {
            // color = vec3(0.1, 0.5, 1.0);
            if (result[4] < nearest_hit[4]) {
                nearest_hit = result;
                color = vec3(1.0, 1.0, 1.0);
            }
        }
    }

    if (nearest_hit[3] == 1) {
        color = vec3(0.1, 0.5, 1.0);
        color = normalize(light - vec3(nearest_hit[0], nearest_hit[1], nearest_hit[2]));
        for (int j = 0; j <= vertices.length(); j += 9) {
            vec4 v0_shadow = vec4(vertices[j    ], vertices[j + 1], vertices[j + 2], 1.0);
            vec4 v1_shadow = vec4(vertices[j + 3], vertices[j + 4], vertices[j + 5], 1.0);
            vec4 v2_shadow = vec4(vertices[j + 6], vertices[j + 7], vertices[j + 8], 1.0);

            v0_shadow = model * v0_shadow;
            v1_shadow = model * v1_shadow;
            v2_shadow = model * v2_shadow;

            float[5] result = ray_triangle_collision(vec3(nearest_hit[0], nearest_hit[1], nearest_hit[2]), normalize(light - vec3(nearest_hit[0], nearest_hit[1], nearest_hit[2])), v0_shadow.xyz, v1_shadow.xyz, v2_shadow.xyz);

            if (result[3] == 1) {
                // color = vec3(0.1, 0.5, 1.0) * 0.8;
                color = vec3(1, 0.5, 0.1);
            }
        }
    }

    fragColor = vec4(color, 1.0);
}




