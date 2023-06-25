#version 460 core

in vec2 position;
in mat4 inverseViewProjection;
in vec3 eye;
in float lightIndex;
in mat4 lightModel;

out vec4 fragColor;

layout (std430, binding = 1) buffer vertex
{ 
    float[] vertices;
};

layout (std430, binding = 2) buffer model
{
    mat4[] model_mats;
};

layout (std430, binding = 3) buffer index
{
    float[] indices;
};


struct hit_t {
    bool exists;
    vec3 pos;
    float t;
    vec3 normal;
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

hit_t ray_triangle_collision(vec3 ray_origin, vec3 ray_direction, vec3 v0, vec3 v1, vec3 v2) {
    hit_t result;

    result.exists = false;

    vec3 edge01 = v1 - v0;
    vec3 edge02 = v2 - v0;

    vec3 normal = normalize(cross(edge01, edge02));
    result.normal = normal;

    float dist = -dot(normal, v0);

    float parallelism = dot(normal, ray_direction);

    if (parallelism != 0.0) {
        result.t = -(dot(normal, ray_origin) + dist) / parallelism;

        if (result.t > 0.0) {
            result.pos = ray_origin + (result.t * ray_direction);

            if (inside_outside_test(v0, v1, v2, result.pos, normal) == 1.0) {
                result.exists = true;
            }
        }
    }

    return(result);
}




void main() {
    // color vector
    vec3 color = vec3(0.1, 0.1, 0.1);

    // FIRST RAY CALCULATION (geometry)

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
    
    // view ray
    vec3 origin = eye;
    vec3 direction = normalize(far.xyz - near.xyz);

    // return value of the collision
    hit_t nearest_hit;
    nearest_hit.exists = false;
    // initial distance from the hit
    nearest_hit.t = 9999999;

    // index variable to keep track of what model we're rendering
    int model_index = 0;
    // counter of vertices inside the model
    float vertex_index = 0;

    model_index = 0;
    vertex_index = 0;

    // for (int k = 0; k <= vertices.length(); k += 3, vertex_index += 3) {
    //     if (vertex_index >= indices[model_index]) {
    //         vertex_index = 0;
    //         model_index += 1;
    //     }

    //     vec4 v = vec4(vertices[k], vertices[k+1], vertices[k+2], 1.0);
        
    //     v = model_mats[model_index] * v;

    //     vertices[k]   = v.x;
    //     vertices[k+1] = v.y;
    //     vertices[k+2] = v.z;
    // }


    for (int i = 0; i <= vertices.length(); i += 9, vertex_index += 9) {

        if (vertex_index >= indices[model_index]) {
            vertex_index = 0;
            model_index += 1;
        }

        vec4 v0 = vec4(vertices[i    ], vertices[i + 1], vertices[i + 2], 1.0);
        vec4 v1 = vec4(vertices[i + 3], vertices[i + 4], vertices[i + 5], 1.0);
        vec4 v2 = vec4(vertices[i + 6], vertices[i + 7], vertices[i + 8], 1.0);

        v0 = model_mats[model_index] * v0;
        v1 = model_mats[model_index] * v1;
        v2 = model_mats[model_index] * v2;

        hit_t hit = ray_triangle_collision(origin, direction, v0.xyz, v1.xyz, v2.xyz);

        if (hit.exists) {
            if (hit.t < nearest_hit.t) {
                nearest_hit = hit;
            }
        }
    }

    if (nearest_hit.exists) {
        color = vec3(0.1, 0.5, 1.0);

        model_index = 0;
        vertex_index = 0;

        for (int j = 0; j <= vertices.length(); j += 9, vertex_index += 9) {
            if (vertex_index >= indices[model_index]) {
                vertex_index = 0;
                model_index += 1;
            }

            if (model_index == lightIndex) {
                break;
            }

            vec4 v0_shadow = vec4(vertices[j    ], vertices[j + 1], vertices[j + 2], 1.0);
            vec4 v1_shadow = vec4(vertices[j + 3], vertices[j + 4], vertices[j + 5], 1.0);
            vec4 v2_shadow = vec4(vertices[j + 6], vertices[j + 7], vertices[j + 8], 1.0);

            v0_shadow = model_mats[model_index] * v0_shadow;
            v1_shadow = model_mats[model_index] * v1_shadow;
            v2_shadow = model_mats[model_index] * v2_shadow;

            float t = 0.00001;

            origin = vec3(nearest_hit.pos + nearest_hit.normal * t);

            hit_t hit = ray_triangle_collision(origin, normalize(vec4(lightModel * vec4(0, 0, 0, 1)).xyz - origin), v0_shadow.xyz, v1_shadow.xyz, v2_shadow.xyz);

            if (hit.exists) {
                color = vec3(0.1, 0.5, 1.0) * 0.5;
            }
        }
    }

    fragColor = vec4(color, 1.0);
}




