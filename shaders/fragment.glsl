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

layout (std430, binding = 4) buffer color
{
    float[] colors;
};

layout (std430, binding = 5) buffer normal
{
    float[] normals;
};


struct hit_t {
    bool exists;
    vec3 pos;
    float t;
    vec3 normal;
    int index;
};


vec3 camera_ray_direction(vec2 pixel, mat4 inverseVP) {
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

    return(normalize(far.xyz - near.xyz));
}

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

hit_t ray_triangle_collision(vec3 ray_origin, vec3 ray_direction, vec3 v0, vec3 v1, vec3 v2, vec3 normal) {
    hit_t result;

    result.exists = false;

    vec3 edge01 = v1 - v0;
    vec3 edge02 = v2 - v0;

    vec3 normal_ = normalize(cross(edge01, edge02));
    result.normal = normal_;

    float dist = -dot(normal, v0);

    float parallelism = dot(normal, ray_direction);

    if (parallelism != 0.0) {
        result.t = -(dot(normal, ray_origin) + dist) / parallelism;

        if (result.t > 0.0) {
            result.pos = ray_origin + (result.t * ray_direction);

            if (inside_outside_test(v0, v1, v2, result.pos, normal_) == 1.0) {
                result.exists = true;
            }
        }
    }

    return(result);
}


hit_t calculate_ray(vec3 ray_origin, vec3 ray_direction, bool shadow) {
    // return value of the collision
    hit_t nearest_hit;
    nearest_hit.exists = false;
    // initial distance from the hit
    nearest_hit.t = 9999999;

    // index variable to keep track of what model we're rendering
    int model_index = 0;
    // counter of vertices inside the model
    float vertex_index = 0;
    int normal_index = 0;

    for (int i = 0; i <= vertices.length(); i += 9, vertex_index += 9, normal_index += 3) {

        if (vertex_index >= indices[model_index]) {
            vertex_index = 0;
            model_index += 1;
        }

        if (shadow) {
            if (model_index == lightIndex) {
                break;
            }
        }

        vec4 v0 = vec4(vertices[i    ], vertices[i + 1], vertices[i + 2], 1.0);
        vec4 v1 = vec4(vertices[i + 3], vertices[i + 4], vertices[i + 5], 1.0);
        vec4 v2 = vec4(vertices[i + 6], vertices[i + 7], vertices[i + 8], 1.0);
        vec4 normal = vec4(normals[normal_index], normals[normal_index + 1], normals[normal_index + 2], 0.0);

        v0 = model_mats[model_index] * v0;
        v1 = model_mats[model_index] * v1;
        v2 = model_mats[model_index] * v2;

        hit_t hit = ray_triangle_collision(ray_origin, ray_direction, v0.xyz, v1.xyz, v2.xyz, normal.xyz);
        hit.index = model_index;


        if (hit.exists) {
            if (shadow) {
                return(hit);
            }

            if (hit.t < nearest_hit.t) {
                nearest_hit = hit;
            }
        }
    }

    return(nearest_hit);
}

void main() {
    // color vector
    vec3 color = vec3(0.1, 0.4, 1.0);

    // FIRST RAY CALCULATION (geometry)
    
    // view ray
    vec3 origin = eye;
    vec3 direction = camera_ray_direction(position.xy, inverseViewProjection);

    hit_t primary_hit = calculate_ray(origin, direction, false);

    if (!primary_hit.exists) {
        fragColor = vec4(color, 1.0);
        return;
    }

    color = vec3(colors[primary_hit.index * 3], colors[primary_hit.index * 3 + 1], colors[primary_hit.index * 3 + 2]);

    float t = 0.1;

    // REFLECTION PASS
    origin = primary_hit.pos + primary_hit.normal * t;
    direction = normalize(direction - 2 * primary_hit.normal * dot(direction, primary_hit.normal));

    hit_t reflection_hit = calculate_ray(origin, direction, false);

    if (reflection_hit.exists) {
        origin = reflection_hit.pos + reflection_hit.normal * t;
        direction = origin - vec4(lightModel * vec4(0, 0, 0, 1)).xyz;

        hit_t reflection_shadow_hit = calculate_ray(origin, direction, true);

        if (reflection_shadow_hit.exists) {
            color += vec3(colors[reflection_hit.index * 3], colors[reflection_hit.index * 3 + 1], colors[reflection_hit.index * 3 + 2]) * 0.5;
            // color = normalize(color);
        } else {
            color += vec3(colors[reflection_hit.index * 3], colors[reflection_hit.index * 3 + 1], colors[reflection_hit.index * 3 + 2]);
            // color = normalize(color);
        }  
    }

    // SHADOW PASS
    origin = vec3(primary_hit.pos + primary_hit.normal * t);
    direction = vec4(lightModel * vec4(0, 0, 0, 1)).xyz - origin;
    
    hit_t shadow_hit = calculate_ray(origin, direction, true);

    if (shadow_hit.exists && shadow_hit.t <= 1) {
        color *= 0.5;
    }

    fragColor = vec4(color, 1.0);
}




