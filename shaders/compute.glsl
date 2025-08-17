#version 430

// custom structs
// ------------------------------------------------------- //
// Struct for a node of the BVH
struct BVHNode {
    vec3 bbox_min;
    int left;
    vec3 bbox_max;
    int right;
    int prim_idx;
    int prim_type;
};

// Struct for a sphere object to render
struct Sphere {
    vec3 center;
    float radius;
    int material;
};

// Struct for a plane object to render
struct Plane {
    vec3 center;
    int material;
    vec3 normal;
};

struct Triangle {
    vec3 v0;
    int material;
    vec3 v1;
    vec3 v2;
};

struct hit_t {
    bool exists;
    vec3 pos;
    float t;
    vec3 normal;
    int index;
    int primitive;
    int material_index;
    vec3 color;
};

// Ray struct
struct Ray {
    vec3 origin;
    vec3 direction;
    vec3 inv_direction;
};

// input-output
// ------------------------------------------------------- //
// position information
layout (local_size_x = 8, local_size_y = 4, local_size_z = 1) in;

// texture to write to
layout (rgba32f, binding = 0) uniform image2D img_output;

#define SIZE 64

// SSBOs and UBOs
layout (std430, binding = 1) buffer vertex { 
    float[] vertices;
};
layout (std140, binding = 2) uniform model {
    mat4[SIZE] model_mats;
};
layout (std430, binding = 3) buffer index {
    float[] indices;
};
layout (binding = 4) uniform normal {
    float[SIZE] mesh_normals;
};
layout (std430, binding = 5) buffer sphere {
    Sphere[] spheres;
};
layout (std430, binding = 6) buffer plane {
    Plane[] planes;
};
layout (binding = 7) uniform box {
    float[SIZE] boxes;
};
layout (std430, binding = 8) buffer bounding_box {
    float[] bounding_boxes;
};
layout (std430, binding = 9) buffer material {
    float[] materials;
};
layout (binding = 10) uniform mesh_material_index {
    float[SIZE] mesh_material_indices;
};
layout(std430, binding = 11) buffer BVHNodes {
    BVHNode nodes[];
};
layout(std430, binding = 12) buffer triangle {
    Triangle triangles[];
};

// uniforms
uniform mat4 inverse_view_projection;
uniform vec3 eye;
uniform float time;
uniform float bounces;
uniform vec3 camera_up;
uniform vec3 camera_right;
uniform vec3 camera_forward;

// constants
// ------------------------------------------------------- //
#define TRIANGLE 0
#define SPHERE 1
#define PLANE 2
#define BOX 3

#define SHINE_OFFSET 3
#define MATERIAL_SIZE 12

const float c_pi = 3.14159265359f;
const float c_twopi = 2.0f * c_pi;

const float HCV_EPSILON = 1e-10;
const float HSL_EPSILON = 1e-10;
const float HCY_EPSILON = 1e-10;

const float SRGB_GAMMA = 1.0 / 2.2;
const float SRGB_INVERSE_GAMMA = 2.2;
const float SRGB_ALPHA = 0.055;


// function declarations
// ------------------------------------------------------- //

// utils
float map(float, float, float, float, float);
float random(inout uint);
vec3 random_unit_vector(inout uint);
vec2 random_point_circle(inout uint);
uint wang_hash(inout uint seed);

// color correction
float linear_to_srgb(float);
float srgb_to_linear(float);
vec3 rgb_to_srgb(vec3);
vec3 srgb_to_rgb(vec3);

// material functions
vec3 get_color(hit_t);
vec4 get_emission(hit_t);
float get_smoothness(hit_t);
vec4 get_albedo(hit_t);

// ray calculation
hit_t calculate_ray(vec3, vec3, bool);
vec3 camera_ray_direction(vec2, mat4);
hit_t find_nearest_hit(hit_t, hit_t, hit_t, hit_t);

// triangles
hit_t ray_triangle_collision(vec3, vec3, vec3, vec3, vec3, vec3);
float inside_outside_test(vec3, vec3, vec3, vec3, vec3);
hit_t calculate_triangles(vec3, vec3, bool, float);

// spheres
hit_t ray_sphere_collision(vec3, vec3, vec3, float);
hit_t calculate_spheres(vec3, vec3, bool, float);

// planes
hit_t ray_plane_collision(vec3, vec3, vec3, vec3);
hit_t calculate_planes(vec3, vec3, bool, float);

// boxes
hit_t ray_box_collision(vec3, vec3, vec3, vec3);
hit_t calculate_boxes(vec3, vec3, bool, float);

// Ray-AABB intersection (slab method)
bool intersectAABB(vec3 rayOrig, vec3 rayDirInv, vec3 bboxMin, vec3 bboxMax, out float tmin, out float tmax) {
    vec3 t1 = (bboxMin - rayOrig) * rayDirInv;
    vec3 t2 = (bboxMax - rayOrig) * rayDirInv;

    vec3 tminVec = min(t1, t2);
    vec3 tmaxVec = max(t1, t2);

    tmin = max(max(tminVec.x, tminVec.y), tminVec.z);
    tmax = min(min(tmaxVec.x, tmaxVec.y), tmaxVec.z);

    return tmax >= max(tmin, 0.0);
}

// BVH traversal returns closest hit
hit_t traverse_bvh(Ray ray) {
    int stack[64];   // traversal stack
    int stackPtr = 0;
    stack[stackPtr++] = 0; // start from root

    hit_t hit;
    hit.exists = false;

    hit.t = 1e20;
    hit.color = vec3(0.0, 0.0, 0.0);
    // hit.prim_idx = -1;

    hit.color = vec3(0.0, 0.0, 0.0);

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        BVHNode node = nodes[nodeIdx];

        float tmin, tmax;
        if (!intersectAABB(ray.origin, ray.inv_direction, node.bbox_min, node.bbox_max, tmin, tmax))
            continue;

        hit.color += vec3(0.01, 0.01, 0.01);

        if (node.prim_idx >= 0) {
            if (node.prim_type == 0) {
                Triangle tri = triangles[node.prim_idx];

                hit_t tri_hit = ray_triangle_collision(ray.origin, ray.direction, tri.v0, tri.v1, tri.v2, vec3(0, 0, 0));

                if (tri_hit.exists) {
                    if (tri_hit.t < hit.t) {
                        tri_hit.color = hit.color;
                        hit = tri_hit;
                        hit.color = vec3(1.0, 1.0, 1.0);
                        hit.material_index = tri.material;
                        // hit.prim_idx = node.prim_idx;
                    }
                }
            } else {
                // hit.t = tmin;
                // hitFound = true;
                // // Leaf node - test sphere
                Sphere sph = spheres[node.prim_idx];

                hit_t sphere_hit = ray_sphere_collision(ray.origin, ray.direction, sph.center, sph.radius);

                if (sphere_hit.exists) {
                    if (sphere_hit.t < hit.t) {
                        sphere_hit.color = hit.color;
                        hit = sphere_hit;
                        hit.color = vec3(1.0, 1.0, 1.0);
                        hit.material_index = sph.material;
                        // hit.prim_idx = node.prim_idx;
                    }
                }
            }
        } else {
            // Internal node - push children
            if (node.left >= 0) stack[stackPtr++] = node.left;
            if (node.right >= 0) stack[stackPtr++] = node.right;
        }
    }

    // return hitFound;
    return hit;
}

hit_t calculate_ray_bvh(Ray ray) {
    hit_t primitives_hit = traverse_bvh(ray);

    hit_t plane_hit = calculate_planes(ray.origin, ray.direction, false, 10000);

    // hit_t triangle_hit = calculate_triangles(ray.origin, ray.direction, false, 10000);
    
    //hit_t box_hit = calculate_boxes(ray_origin, ray_direction, shadow, 10000);

    hit_t nearest_hit = find_nearest_hit(primitives_hit, plane_hit, plane_hit, plane_hit);

    return nearest_hit;
}

// function definition
// ------------------------------------------------------- //

// main function
void main() {
    // get the position of the texel to draw based on the compute shader dispatch coordinate
    ivec2 texel_coord = ivec2(gl_GlobalInvocationID.xy);

    // get the size of the texture to write to
    ivec2 dims = imageSize(img_output);

    // calculate the position vector
    vec2 position = vec2(
        float(texel_coord.x * 2 - dims.x) / dims.x,
        float(texel_coord.y * 2 - dims.y) / dims.y
    );

    // calculate an rng value
    uint rng_state = uint(uint(texel_coord.x) * uint(1973) + uint(texel_coord.y) * uint(9277) + uint(time * 1000) * uint(26699)) | uint(1);
    
    // use the rng value to calculate a jitter to apply to the ray
    // vec2 jitter = random_point_circle(rng_state) / dims.x;
    vec2 jitter = random_point_circle(rng_state) / vec2(dims);


    // color vector
    vec3 void_color = srgb_to_rgb(vec3(0.5, 0.5, 1.0));
    vec3 color = vec3(1.0, 1.0, 1.0);


    Ray ray;
    
    // store the eye position as the origin of the ray
    ray.origin = eye;

    // calculate the direction of the ray based on the camera matrix
    ray.direction = camera_ray_direction(position.xy + jitter, inverse_view_projection);

    ray.inv_direction = 1 / ray.direction;

    // declare the result of the ray hit
    hit_t hit;

    // prepare the amount of light captured by the ray
    vec3 light = vec3(0);

    // iterate through the number of bounces
    for (int i = 0; i < int(bounces); i++) {
        // do the ray intersection calculationù
        hit = calculate_ray_bvh(ray);

        // if the ray didn't hit anything, break out of the loop
        if (!hit.exists) {
            light = vec3(0.2, 0.5, 1.0);
            break;
        }

        // get the emission from the hit response
        vec4 emission = get_emission(hit);
        
        // get the emitted light by the color multiplied by the strength
        vec3 emitted_light = emission.xyz * emission.w;

        // add the emitted light multiplied by the color to get the total light
        light += emitted_light * color;

        // if we reached the last bounce, we break early
        if (i == int(bounces) + 1) {
          break;
        }
       
        // calculate the origin of the next ray as the last hit offset slightly on the normal
        ray.origin = vec3(hit.pos + hit.normal * 0.001);
        
        // pick a random direction from the normal to calculate the next ray for diffuse lighting 
        vec3 diffuse_direction = normalize(hit.normal + random_unit_vector(rng_state));
        // calculate the reflection ray from the surface to get the specular lighting
        vec3 specular_direction = normalize(ray.direction - 2 * hit.normal * dot(ray.direction, hit.normal));

        // extract the albedo color from the hit data
        vec4 specular = get_albedo(hit);

        // get the chance of reflecting ?
        bool albedo_chance = specular.w >= random(rng_state);

        // calculate the direction of the bounce by mixing the specular and diffuse directions with the smoothness and albedo chance
        ray.direction = mix(diffuse_direction, specular_direction, get_smoothness(hit) * int(albedo_chance));

        // if the obtained direction is opposite to the normal, invert it
        if (dot(ray.direction, hit.normal) < 0) {
            ray.direction = -ray.direction;
        }

        ray.inv_direction = 1 / ray.direction;

        // calculate the light strength based on the angle between the normal and the bounce direction
        float light_strength = dot(hit.normal, ray.direction);

        // get the color for the next hit
        color *= mix(get_color(hit), specular.xyz, albedo_chance ? 1.0 : 0.0) * light_strength;
    }

    // store the pixel in the texture
    imageStore(img_output, texel_coord, vec4(rgb_to_srgb(light), 1.0));

    // imageStore(img_output, texel_coord, vec4(hit.color, 1.0));
    // imageStore(img_output, texel_coord, vec4(triangles[0].v2, 1.0));
}




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


vec3 get_color(hit_t target) {
    vec3 color = vec3(0, 0, 0);

    // if (target.primitive == TRIANGLE) {
    //     color = srgb_to_rgb(vec3(mesh_materials  [target.index * MATERIAL_SIZE], mesh_materials  [target.index * MATERIAL_SIZE + 1], mesh_materials  [target.index * MATERIAL_SIZE + 2]));
    // } else if (target.primitive == SPHERE) {
    //     color = srgb_to_rgb(vec3(sphere_materials[target.index * MATERIAL_SIZE], sphere_materials[target.index * MATERIAL_SIZE + 1], sphere_materials[target.index * MATERIAL_SIZE + 2]));
    // } else if (target.primitive == PLANE) {
    //     color = srgb_to_rgb(vec3(plane_materials [target.index * MATERIAL_SIZE], plane_materials [target.index * MATERIAL_SIZE + 1], plane_materials [target.index * MATERIAL_SIZE + 2]));
    // } else {
    //     color = srgb_to_rgb(vec3(box_materials   [target.index * MATERIAL_SIZE], box_materials   [target.index * MATERIAL_SIZE + 1], box_materials   [target.index * MATERIAL_SIZE + 2]));
    // }

    if (target.primitive == TRIANGLE) {
        color = srgb_to_rgb(vec3(materials[int(mesh_material_indices[target.index]) * MATERIAL_SIZE], materials[int(mesh_material_indices[target.index]) * MATERIAL_SIZE + 1], materials[int(mesh_material_indices[target.index]) * MATERIAL_SIZE + 2]));
    } else {
        color = srgb_to_rgb(vec3(materials[target.material_index * MATERIAL_SIZE], materials[target.material_index * MATERIAL_SIZE + 1], materials[target.material_index * MATERIAL_SIZE + 2]));
    }

    return(color);
}

vec4 get_emission(hit_t target) {
    vec4 emission = vec4(0, 0, 0, 0);

    if (target.primitive == TRIANGLE) {
        emission = vec4(srgb_to_rgb(vec3(materials[int(mesh_material_indices[target.index]) * MATERIAL_SIZE + 3], materials[int(mesh_material_indices[target.index]) * MATERIAL_SIZE + 4], materials[int(mesh_material_indices[target.index]) * MATERIAL_SIZE + 5])), materials[int(mesh_material_indices[target.index]) * MATERIAL_SIZE + 6]);
    } else {
        emission = vec4(srgb_to_rgb(vec3(materials[target.material_index * MATERIAL_SIZE + 3], materials[target.material_index * MATERIAL_SIZE + 4], materials[target.material_index * MATERIAL_SIZE + 5])), materials[target.material_index * MATERIAL_SIZE + 6]);
    }

    return(emission);
}

float get_smoothness(hit_t target) {
    float smoothness = 0;

    if (target.primitive == TRIANGLE) {
        smoothness = materials[int(mesh_material_indices[target.index]) * MATERIAL_SIZE + 7];
    } else {
        smoothness = materials[target.material_index * MATERIAL_SIZE + 7];
    }

    return(smoothness);
}

vec4 get_albedo(hit_t target) {
    vec4 albedo = vec4(0);

    if (target.primitive == TRIANGLE) {
        albedo = vec4(materials[int(mesh_material_indices[target.index]) * MATERIAL_SIZE + 8], materials[int(mesh_material_indices[target.index]) * MATERIAL_SIZE + 9], materials[int(mesh_material_indices[target.index]) * MATERIAL_SIZE + 10], materials[int(mesh_material_indices[target.index]) * MATERIAL_SIZE + 11]);
    } else {
        albedo = vec4(materials[target.material_index * MATERIAL_SIZE + 8], materials[target.material_index * MATERIAL_SIZE + 9], materials[target.material_index * MATERIAL_SIZE + 10], materials[target.material_index * MATERIAL_SIZE + 11]);
    }

    return(albedo);
}

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

uint wang_hash(inout uint seed) {
    seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> 4);
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> 15);
    return seed;
}

float random(inout uint state) {
    return float(wang_hash(state)) / 4294967296.0;
}

vec3 random_unit_vector(inout uint state) {
    float z = random(state) * 2.0f - 1.0f;
    float a = random(state) * c_twopi;
    float r = sqrt(1.0f - z * z);
    float x = r * cos(a);
    float y = r * sin(a);
    return vec3(x, y, z);
}

vec2 random_point_circle(inout uint state) {
    float angle = random(state) * 2 * c_pi;
    vec2 point_on_circle = vec2(cos(angle), sin(angle));
    return(point_on_circle * sqrt(random(state)));
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

    if (-denom > 0.0000001) {
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

    for (int i = 0; i <= planes.length(); i++) {
        vec3 position = vec3(planes[i].center.x, planes[i].center.y, planes[i].center.z);
        vec3 normal = vec3(planes[i].normal.x, planes[i].normal.y, planes[i].normal.z);

        hit_t hit = ray_plane_collision(ray_origin, ray_direction, position, normal);
        hit.index = i;
        hit.material_index = int(planes[i].material);

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

    for (int i = 0; i <= spheres.length(); i++) {
        vec3 center = spheres[i].center;
        float radius = spheres[i].radius;

        hit_t hit = ray_sphere_collision(ray_origin, ray_direction, center, radius);
        hit.index = i;
        hit.material_index = int(spheres[i].material);

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

hit_t ray_triangle_collision(vec3 ray_origin, vec3 ray_direction, vec3 v0, vec3 v1, vec3 v2, vec3 normal) {
    hit_t result;

    result.exists = false;
    result.primitive = TRIANGLE;

    vec3 edge01 = v1 - v0;
    vec3 edge02 = v2 - v0;

    vec3 normal_ = normalize(cross(edge01, edge02));
    result.normal = normal_;

    float dist = -dot(normal_, v0);

    float parallelism = dot(normal_, ray_direction);

    if (parallelism != 0.0) {
        result.t = -(dot(normal_, ray_origin) + dist) / parallelism;

        if (result.t > 0.0) {
            result.pos = ray_origin + (result.t * ray_direction);

            if (inside_outside_test(v0, v1, v2, result.pos, normal_) == 1.0) {
                result.exists = true;
            }
        }
    }

    return(result);
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

    for (int i = 0; i <= boxes.length(); i += 7) {
        vec3 b0 = vec3(boxes[i], boxes[i + 1], boxes[i + 2]);
        vec3 b1 = vec3(boxes[i + 3], boxes[i + 4], boxes[i + 5]);

        hit_t hit = ray_box_collision(ray_origin, ray_direction, b0, b1);
        hit.index = int(i / 7);
        hit.material_index = int(boxes[i + 6]);

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


hit_t calculate_triangles(vec3 ray_origin, vec3 ray_direction, bool shadow, float max_t) {
    hit_t nearest_hit;
    nearest_hit.exists = false;
    nearest_hit.t = 9999999;

    for (int i = 0; i <= triangles.length(); i++) {
        hit_t hit = ray_triangle_collision(ray_origin, ray_direction, triangles[i].v0, triangles[i].v1, triangles[i].v2, vec3(0, 0, 0));
        hit.index = i;
        hit.material_index = int(triangles[i].material);

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

    //hit_t triangle_hit = calculate_triangles(ray_origin, ray_direction, shadow, 10000);

    hit_t sphere_hit = calculate_spheres(ray_origin, ray_direction, shadow, 10000);

    hit_t plane_hit = calculate_planes(ray_origin, ray_direction, shadow, 10000);
    
    //hit_t box_hit = calculate_boxes(ray_origin, ray_direction, shadow, 10000);

    hit_t nearest_hit = find_nearest_hit(sphere_hit, plane_hit, sphere_hit, plane_hit);


    return(nearest_hit);
    //return(sphere_hit);
}


