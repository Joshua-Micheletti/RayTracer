#version 430

layout (local_size_x = 8, local_size_y = 4, local_size_z = 1) in;

layout (rgba32f, binding = 0) uniform image2D imgOutput;


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

uniform mat4 inverseViewProjection;
uniform vec3 eye;
uniform float lightIndex;
uniform mat4 lightModel;

#define TRIANGLE 0
#define SPHERE 1
#define PLANE 2
#define BOX 3

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
    vec4 screenSpaceFar = vec4(pixel.xy, 1.0, 1.0);
    // coordinate of origin ray in screen space
    vec4 screenSpaceNear = vec4(pixel.xy, 0.0, 1.0);
    
    // coordinate of end ray in world space
    vec4 far = inverseVP * screenSpaceFar;
    far /= far.w;
    // coordinate of origin ray in world space
    vec4 near = inverseVP * screenSpaceNear;
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

hit_t ray_plane_collision(vec3 ray_origin, vec3 ray_direction, vec3 position, vec3 normal) {
    hit_t hit;
    hit.exists = false;
    hit.primitive = PLANE;

    float denom = dot(normal, ray_direction);
    float t;

    if (abs(denom) > 0.0000001) {
        t = dot(position - ray_origin, normal) / denom;
        
        if (t > 0) {
            hit.exists = true;
            hit.t = t;
            hit.normal = normal;
            hit.pos = ray_origin + ray_direction * t;
        }
    }

    return(hit);
}

hit_t calculate_planes(vec3 ray_origin, vec3 ray_direction, bool shadow, float max_t) {
    hit_t nearest_hit;
    nearest_hit.exists = false;
    nearest_hit.t = 9999999;

    vec3 position = vec3(0, 0, 0);
    vec3 normal = vec3(0, 1, 0);

    hit_t hit = ray_plane_collision(ray_origin, ray_direction, position, normal);

    if (hit.exists) {
        if (shadow) {
            if (hit.t <= max_t) {
                return(hit);
            }
        } else if (hit.t < nearest_hit.t) {
            nearest_hit = hit;
        }
    }

    return(nearest_hit);
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

hit_t calculate_spheres(vec3 ray_origin, vec3 ray_direction, bool shadow, float max_t) {
    hit_t nearest_hit;
    nearest_hit.exists = false;
    nearest_hit.t = 999999;

    vec3 center = vec3(-1.0, 1.0, -1.0);

    float radius = 0.5;

    hit_t hit = ray_sphere_collision(ray_origin, ray_direction, center, radius);

    if (hit.exists) {
        if (shadow) {
            if (hit.t <= max_t) {
                return(hit);
            }
        } else if (hit.t < nearest_hit.t) {
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

hit_t calculate_triangles(vec3 ray_origin, vec3 ray_direction, bool shadow, float max_t) {
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
                if (hit.t <= max_t) {
                    return(hit);
                }
            } else if (hit.t < nearest_hit.t) {
                nearest_hit = hit;
            }
        }
    }

    return(nearest_hit);
}

hit_t ray_box_collision(vec3 ray_origin, vec3 ray_direction, vec3 b0, vec3 b1) {
    hit_t hit;
    hit.exists = false;
    hit.primitive = BOX;

    vec3 invdir = 1 / ray_direction;

    vec3 tbot = invdir * (b0 - ray_origin);
    vec3 ttop = invdir * (b1 - ray_origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    vec2 t = max(tmin.xx, tmin.yz);
    float t0 = max(t.x, t.y);
    t = min(tmax.xx, tmax.yz);
    float t1 = min(t.x, t.y);

    if (t1 <= max(t0, 0.0)) {
        return(hit);
    }

    hit.exists = true;
    hit.t = t0;
    hit.pos = ray_origin + ray_direction * t0;

    vec3 near = b0;
    vec3 far = b1;

    float epsi = 0.00001;

    if (abs(hit.pos.x - near.x) < epsi) {
        hit.normal = vec3(-1, 0, 0);
    } else if (abs(hit.pos.x - far.x) < epsi) {
        hit.normal = vec3(1, 0, 0);
    } else if (abs(hit.pos.y - near.y) < epsi) {
        hit.normal = vec3(0, -1, 0);
    } else if (abs(hit.pos.y - far.y) < epsi) {
        hit.normal = vec3(0, 1, 0);
    } else if (abs(hit.pos.z - near.z) < epsi) {
        hit.normal = vec3(0, 0, -1);
    } else if (abs(hit.pos.z - far.z) < epsi) {
        hit.normal = vec3(0, 0, 1);
    }


    return(hit);
}

hit_t calculate_boxes(vec3 ray_origin, vec3 ray_direction, bool shadow, float max_t) {
    hit_t nearest_hit;
    nearest_hit.exists = false;
    nearest_hit.t = 9999999;

    vec3 b0 = vec3(-2, 2, 2);
    vec3 b1 = vec3(-1, 3, 3);

    hit_t hit = ray_box_collision(ray_origin, ray_direction, b0, b1);

    if (hit.exists) {
        if (shadow) {
            if (hit.t <= max_t) {
                return(hit);
            }
        } else if (hit.t < nearest_hit.t) {
            nearest_hit = hit;
        }
    }

    return(nearest_hit);
}

hit_t find_nearest_hit(hit_t h0, hit_t h1, hit_t h2, hit_t h3) {
    hit_t nearest_hit;
    nearest_hit.exists = false;

    nearest_hit = h0;

    if (!nearest_hit.exists && h1.exists) {
        nearest_hit = h1;
    } else if (nearest_hit.exists && h1.exists && h1.t < nearest_hit.t) {
        nearest_hit = h1;
    }

    if (!nearest_hit.exists && h2.exists) {
        nearest_hit = h2;
    } else if (nearest_hit.exists && h2.exists && h2.t < nearest_hit.t) {
        nearest_hit = h2;
    }

    if (!nearest_hit.exists && h3.exists) {
        nearest_hit = h3;
    } else if (nearest_hit.exists && h3.exists && h3.t < nearest_hit.t) {
        nearest_hit = h3;
    }

    return(nearest_hit);
}

hit_t calculate_ray(vec3 ray_origin, vec3 ray_direction, bool shadow) {

    hit_t triangle_hit = calculate_triangles(ray_origin, ray_direction, shadow, 10000);

    hit_t sphere_hit = calculate_spheres(ray_origin, ray_direction, shadow, 10000);

    hit_t plane_hit = calculate_planes(ray_origin, ray_direction, shadow, 10000);

    hit_t box_hit = calculate_boxes(ray_origin, ray_direction, shadow, 10000);


    hit_t nearest_hit = find_nearest_hit(triangle_hit, sphere_hit, plane_hit, box_hit);


    return(nearest_hit);
}

hit_t calculate_shadow_ray(vec3 ray_origin, vec3 ray_direction, vec3 light_position) {
    float t_to_light = (light_position.x - ray_origin.x) / ray_direction.x;

    hit_t triangle_hit = calculate_triangles(ray_origin, ray_direction, true, t_to_light);

    if (triangle_hit.exists) {
        return(triangle_hit);
    }

    hit_t sphere_hit = calculate_spheres(ray_origin, ray_direction, true, t_to_light);

    if (sphere_hit.exists) {
        return(sphere_hit);
    }

    hit_t plane_hit = calculate_planes(ray_origin, ray_direction, true, t_to_light);

    if (plane_hit.exists) {
        return(plane_hit);
    }

    hit_t box_hit = calculate_boxes(ray_origin, ray_direction, true, t_to_light);

    if (box_hit.exists) {
        return(box_hit);
    }

    hit_t none;
    none.exists = false;
    return(none);
}


void main() {
    // vec4 position = vec4(0.0, 0.0, 0.0, 1.0);
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
    
    vec2 position = vec2(0.0, 0.0);

    ivec2 dims = imageSize(imgOutput);

    position.x = (float(texelCoord.x * 2 - dims.x) / dims.x);
    position.y = (float(texelCoord.y * 2 - dims.y) / dims.y);

    // color vector
    vec3 void_color = vec3(0.1, 0.4, 1.0);
    vec3 color = vec3(0.1, 0.4, 1.0);
    vec3 sphere_color = vec3(0.5, 0.5, 0.5);
    vec3 plane_color = vec3(0.1, 1.0, 0.4);
    vec3 box_color = vec3(1.0, 0.1, 1.0);
    
    vec3 origin = eye;
    vec3 direction = camera_ray_direction(position.xy, inverseViewProjection);

    hit_t primary_hit = calculate_ray(origin, direction, false);

    if (!primary_hit.exists) {
        imageStore(imgOutput, texelCoord, vec4(color, 1.0));
        return;
    }

    if (primary_hit.primitive == TRIANGLE) {
        color = vec3(colors[primary_hit.index * 3], colors[primary_hit.index * 3 + 1], colors[primary_hit.index * 3 + 2]);
    } else if (primary_hit.primitive == SPHERE) {
        color = sphere_color;
    } else if (primary_hit.primitive == PLANE) {
        color = plane_color;
    } else {
        color = box_color;
    }

    float t = 0.0001;

    // REFLECTION PASS
    origin = primary_hit.pos + primary_hit.normal * t;
    direction = normalize(direction - 2 * primary_hit.normal * dot(direction, primary_hit.normal));

    hit_t reflection_hit = calculate_ray(origin, direction, false);

    if (reflection_hit.exists) {
        vec3 light_position = vec4(lightModel * vec4(0, 0, 0, 1)).xyz;
        origin = reflection_hit.pos + reflection_hit.normal * t;
        direction = normalize(light_position - origin);

        hit_t reflection_shadow_hit = calculate_shadow_ray(origin, direction, light_position);

        if (reflection_shadow_hit.exists) {
            if (reflection_hit.primitive == TRIANGLE) {
                color += vec3(colors[reflection_hit.index * 3], colors[reflection_hit.index * 3 + 1], colors[reflection_hit.index * 3 + 2]) * 0.5;
            } else if (reflection_hit.primitive == SPHERE) {
                color += sphere_color * 0.5;
            } else if (reflection_hit.primitive == PLANE) {
                color += plane_color * 0.5;
            } else {
                color += box_color * 0.5;
            }

        } else {
            if (reflection_hit.primitive == TRIANGLE) {
                color += vec3(colors[reflection_hit.index * 3], colors[reflection_hit.index * 3 + 1], colors[reflection_hit.index * 3 + 2]);
            } else if (reflection_hit.primitive == SPHERE) {
                color += sphere_color;
            } else if (reflection_hit.primitive == PLANE) {
                color += plane_color;
            } else {
                color += box_color;
            }
        }  
    } else {
        color += void_color * 0.5;
    }

    // SHADOW PASS
    vec3 light_position = vec4(lightModel * vec4(0, 0, 0, 1)).xyz;

    origin = vec3(primary_hit.pos + primary_hit.normal * t);
    direction = normalize(light_position - origin);
    
    hit_t shadow_hit = calculate_shadow_ray(origin, direction, light_position);

    float t_to_light = (light_position.x - origin.x) / direction.x;

    if (shadow_hit.exists) {
        color *= 0.5;
    }


    imageStore(imgOutput, texelCoord, vec4(color, 1.0));
}