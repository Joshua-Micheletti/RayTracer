#version 460 core

in vec2 position;
in mat4 inverseViewProjection;
in vec3 eye;
in float lightIndex;
in mat4 lightModel;

out vec4 fragColor;

#define TRIANGLE 0
#define SPHERE 1



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
    int primitive;
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

hit_t ray_sphere_collision(vec3 ray_origin, vec3 ray_direction, vec3 sphere_center, float sphere_radius) {
        hit_t hit;
        hit.exists = false;

        float t0, t1; // solutions for t if the ray intersects

        // geometric solution
        vec3 L = sphere_center - ray_origin;
        float tca = dot(L, ray_direction);
        // if (tca < 0) return hit;
        float d2 = dot(L, L) - tca * tca;
        float radius2 = sphere_radius * sphere_radius;
        if (d2 > radius2) return(hit);
        float thc = sqrt(radius2 - d2);
        t0 = tca - thc;
        t1 = tca + thc;

        if (t0 > t1) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        if (t0 < 0) {
            t0 = t1; // if t0 is negative, let's use t1 instead
            if (t0 < 0) return(hit); // both t0 and t1 are negative
        }

        hit.exists = true;
        hit.t = t0;

        hit.pos = ray_origin + ray_direction * hit.t;
        hit.normal = normalize(hit.pos - sphere_center);
        hit.primitive = SPHERE;
        

        return(hit);
}

hit_t calculate_spheres(vec3 ray_origin, vec3 ray_direction, bool shadow) {
    hit_t nearest_hit;
    nearest_hit.exists = false;
    nearest_hit.t = 999999;

    vec3 center = vec3(-1.0, -1.0, -1.0);

    float radius = 0.5;

    hit_t hit = ray_sphere_collision(ray_origin, ray_direction, center, radius);

    if (hit.exists) {
        if (shadow) {
            if (hit.t <= 1 && hit.t > 0) {
                return(hit);
            }
        }

        else if (hit.t < nearest_hit.t) {
            nearest_hit = hit;
        }
    }

    return(nearest_hit);
}

hit_t ray_triangle_collision(vec3 ray_origin, vec3 ray_direction, vec3 v0, vec3 v1, vec3 v2, vec3 normal) {
    hit_t result;

    result.exists = false;
    result.primitive = TRIANGLE;

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

hit_t calculate_triangles(vec3 ray_origin, vec3 ray_direction, bool shadow) {
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

            else if (hit.t < nearest_hit.t) {
                nearest_hit = hit;
            }
        }
    }

    return(nearest_hit);
}

hit_t calculate_ray(vec3 ray_origin, vec3 ray_direction, bool shadow) {

    hit_t triangle_hit = calculate_triangles(ray_origin, ray_direction, shadow);

    hit_t sphere_hit = calculate_spheres(ray_origin, ray_direction, shadow);


    // if (shadow) {
    //     if (triangle_hit.exists && sphere_hit.exists) {
    //         return(triangle_hit);
    //     }

    //     if (triangle_hit.exists) {
    //         return(triangle_hit);
    //     }

    //     if (sphere_hit.exists) {
    //         return(sphere_hit);
    //     }

    //     return(triangle_hit);
    // }

    if (triangle_hit.exists && sphere_hit.exists) {
        if (triangle_hit.t < sphere_hit.t) {
            return(triangle_hit);
        } else {
            return(sphere_hit);
        }
    }

    if (sphere_hit.exists) {
        return(sphere_hit);
    }

    if (triangle_hit.exists) {
        return(triangle_hit);
    }

    hit_t none;
    none.exists = false;

    return(none);
}

	


void main() {
    // color vector
    vec3 color = vec3(0.1, 0.4, 1.0);
    vec3 sphere_color = vec3(1.0, 0.2, 0.1);

    // FIRST RAY CALCULATION (geometry)
    
    // view ray
    vec3 origin = eye;
    vec3 direction = camera_ray_direction(position.xy, inverseViewProjection);

    // bool hit = false;

    // for (float i = -5; i < 5; i++) {
    //     if (ray_sphere_collision(origin, direction, vec3(i, 0, 0), 0.4)) {
    //         hit = true;
    //     }
    // }

    // if (hit) {
    //     color = vec3(1, 0.2, 0.5);
    // }

    // if (ray_sphere_collision(origin, direction, vec3(0.0), 1.0)) {
    //     color = vec3(1.0, 1.0, 1.0);
    // }


    hit_t primary_hit = calculate_ray(origin, direction, false);

    if (!primary_hit.exists) {
        fragColor = vec4(color, 1.0);
        return;
    }

    if (primary_hit.primitive == TRIANGLE) {
        color = vec3(colors[primary_hit.index * 3], colors[primary_hit.index * 3 + 1], colors[primary_hit.index * 3 + 2]);
    } else {
        color = sphere_color;
    }

    // color = vec3(1.0);

    float t = 0.0001;

    // REFLECTION PASS
    origin = primary_hit.pos + primary_hit.normal * t;
    direction = normalize(direction - 2 * primary_hit.normal * dot(direction, primary_hit.normal));

    hit_t reflection_hit = calculate_ray(origin, direction, false);

    if (reflection_hit.exists) {
        origin = reflection_hit.pos + reflection_hit.normal * t;
        direction = origin - vec4(lightModel * vec4(0, 0, 0, 1)).xyz;

        hit_t reflection_shadow_hit = calculate_ray(origin, direction, true);

        if (reflection_shadow_hit.exists) {
            if (reflection_hit.primitive == TRIANGLE) {
                color += vec3(colors[reflection_hit.index * 3], colors[reflection_hit.index * 3 + 1], colors[reflection_hit.index * 3 + 2]) * 0.5;
            } else {
                color += sphere_color * 0.5;
            }
            // color = normalize(color);
        } else {
            if (reflection_hit.primitive == TRIANGLE) {
                color += vec3(colors[reflection_hit.index * 3], colors[reflection_hit.index * 3 + 1], colors[reflection_hit.index * 3 + 2]);
            } else {
                color += sphere_color;
            }
            
            // color = normalize(color);
        }  
    }

    // SHADOW PASS
    vec3 light_position = vec4(lightModel * vec4(0, 0, 0, 1)).xyz;

    origin = vec3(primary_hit.pos + primary_hit.normal * t);
    direction = normalize(light_position - origin);
    
    hit_t shadow_hit = calculate_ray(origin, direction, true);

    float t_to_light = (light_position.x - origin.x) / direction.x;


    if (shadow_hit.exists && shadow_hit.t <= t_to_light) {
        color *= 0.5;
    }

    fragColor = vec4(color, 1.0);
}




