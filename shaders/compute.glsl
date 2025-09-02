#version 430

// custom structs
// ------------------------------------------------------- //
// Struct for a node of the BVH
struct BVHNode {
    uvec4 aabb_x;
    uvec4 aabb_y;
    uvec4 aabb_z;
    uvec4 metadata;
};

// Struct for a sphere object to render
struct Sphere {
    vec3 center;
    float radius;
    float radius2;
    uint material;
};

// Struct for a plane object to render
struct Plane {
    vec3 center;
    uint material;
    vec3 normal;
};

// Struct for a triangle object to render
struct Triangle {
    vec3 v0;
    uint material;
    vec3 v1;
    vec3 v2;
    vec3 normal;
};

struct Box {
    vec3 p0; 
    uint material;
    vec3 p1;
};

struct Material {
    uint color;
    uint emission;
    uint properties;
};

struct Hit {
    bool exists;
    vec3 pos;
    float t;
    vec3 normal;
    uint material_index;
};

// Ray struct
struct Ray {
    vec3 origin;
    vec3 direction;
    vec3 inv_direction;
    float ior;
};

// input-output
// ------------------------------------------------------- //
// position information
layout (local_size_x = 8, local_size_y = 4, local_size_z = 1) in;

// texture to write to
layout (rgba32f, binding = 0) writeonly uniform image2D img_output;

// SSBOs
layout (std430, binding = 5) readonly buffer sphere {
    Sphere[] spheres;
};
layout (std430, binding = 6) readonly buffer plane {
    Plane[] planes;
};
layout (std430, binding = 7) readonly buffer box {
    Box[] boxes;
};
layout (std430, binding = 9) readonly buffer material_buffer {
    Material[] materials;
};
layout(std430, binding = 11) readonly buffer BVHNodes {
    BVHNode nodes[];
};
layout(std430, binding = 12) readonly buffer triangle {
    Triangle triangles[];
};

// uniforms
uniform mat4 inverse_view_projection;
uniform vec3 eye;
uniform vec3 scene_min;
uniform vec3 scene_extent;
uniform float time;
uniform float bounces;
uniform vec3 camera_up;
uniform vec3 camera_right;
uniform vec3 camera_forward;

// constants
// ------------------------------------------------------- //
#define TRIANGLE 0
#define SPHERE 1
#define BOX 2
#define NO_PRIMITIVE 3

#define NO_INDEX 268435455

#define SHINE_OFFSET 3
#define MATERIAL_SIZE 12
const float INV_UINT_MAX_PLUS_ONE = 1.0 / 4294967296.0;

const float c_pi = 3.14159265359f;
const float c_twopi = 2.0f * c_pi;

const float HCV_EPSILON = 1e-10;
const float HSL_EPSILON = 1e-10;
const float HCY_EPSILON = 1e-10;

const float SRGB_GAMMA = 1.0 / 2.2;
const float SRGB_INVERSE_GAMMA = 2.2;
const float SRGB_ALPHA = 0.055;

const uint INDEX_MASK = 0x3FFFFFFFu;  
const uint TYPE_MASK  = 0x3u;
const uint TYPE_SHIFT = 30u;


// function declarations
// ------------------------------------------------------- //

// utils
float map(float, float, float, float, float);
float random(inout uint);
uint wang_hash(inout uint seed);
vec4 unpackFloat4x8(uint);
uvec2 unpackUint2x16(uint);
vec3 unpackRGB(uint);

// color correction
float linear_to_srgb(float);
float srgb_to_linear(float);
vec3 rgb_to_srgb(vec3);
vec3 srgb_to_rgb(vec3);

// ray calculation
Hit calculate_ray(vec3, vec3, bool);
vec3 camera_ray_direction(vec2, mat4);
Hit find_nearest_hit(Hit, Hit, Hit, Hit);

// triangles
Hit ray_triangle_collision(vec3, vec3, vec3, vec3, vec3, vec3);
Hit calculate_triangles(vec3, vec3, bool, float);

// spheres
Hit ray_sphere_collision(vec3, vec3, vec3, float, float);
Hit calculate_spheres(vec3, vec3, bool, float);

// planes
Hit ray_plane_collision(vec3, vec3, vec3, vec3);
Hit calculate_planes(vec3, vec3, bool, float);

// boxes
Hit ray_box_collision(vec3, vec3, vec3, vec3);
Hit calculate_boxes(vec3, vec3, bool, float);

// Ray-AABB intersection (slab method)
bool intersect_aabb(vec3 rayOrig, vec3 rayDirInv, vec3 bboxMin, vec3 bboxMax, out float tmin) {
    vec3 t1 = (bboxMin - rayOrig) * rayDirInv;
    vec3 t2 = (bboxMax - rayOrig) * rayDirInv;

    vec3 tminVec = min(t1, t2);
    vec3 tmaxVec = max(t1, t2);

    tmin = max(max(tminVec.x, tminVec.y), tminVec.z);
    float tmax = min(min(tmaxVec.x, tmaxVec.y), tmaxVec.z);

    // return step(max(tmin, 0.0), tmax) > 0.0;
    return tmax >= max(tmin, 0.0);
}

bvec4 intersectAABB4(in Ray ray, vec4 minX, vec4 maxX, vec4 minY, vec4 maxY, vec4 minZ, vec4 maxZ,
                     out vec4 tNear, out vec4 tFar)
{
    vec4 t1x = (minX - ray.origin.x) * ray.inv_direction.x;
    vec4 t2x = (maxX - ray.origin.x) * ray.inv_direction.x;
    vec4 tminx = min(t1x, t2x);
    vec4 tmaxx = max(t1x, t2x);

    vec4 t1y = (minY - ray.origin.y) * ray.inv_direction.y;
    vec4 t2y = (maxY - ray.origin.y) * ray.inv_direction.y;
    vec4 tminy = min(t1y, t2y);
    vec4 tmaxy = max(t1y, t2y);

    vec4 t1z = (minZ - ray.origin.z) * ray.inv_direction.z;
    vec4 t2z = (maxZ - ray.origin.z) * ray.inv_direction.z;
    vec4 tminz = min(t1z, t2z);
    vec4 tmaxz = max(t1z, t2z);

    tNear = max(max(tminx, tminy), max(tminz, vec4(0.0)));
    tFar  = min(min(tmaxx, tmaxy), tmaxz);

    return lessThanEqual(tNear, tFar);
}

void sort4(inout float t[4], inout int idx[4], inout bool valid[4]) {
    #define SWAP(i,j) { \
        bool swap = (t[j] < t[i]); \
        float tt = swap ? t[j] : t[i]; t[j] = swap ? t[i] : t[j]; t[i] = tt; \
        int ii = swap ? idx[j] : idx[i]; idx[j] = swap ? idx[i] : idx[j]; idx[i] = ii; \
        bool vv = swap ? valid[j] : valid[i]; valid[j] = swap ? valid[i] : valid[j]; valid[i] = vv; \
    }

    SWAP(0,1); SWAP(2,3);
    SWAP(0,2); SWAP(1,3);
    SWAP(1,2);

    #undef SWAP
}

Hit traverse_bvh(Ray ray) {
    float tMax = 1e38;
    Hit bestHit;
    bestHit.t = tMax;

    uint stack[32];
    uint stackPtr = 0;
    stack[stackPtr++] = 0; // root

    while (stackPtr > 0) {
        uint nodeIdx = stack[--stackPtr];
        BVHNode node = nodes[nodeIdx];

        // Unpack X, Y, Z aabb channels
        uvec4 ax = uvec4(node.aabb_x.x, node.aabb_x.y, node.aabb_x.z, node.aabb_x.w);
        uvec4 ay = uvec4(node.aabb_y.x, node.aabb_y.y, node.aabb_y.z, node.aabb_y.w);
        uvec4 az = uvec4(node.aabb_z.x, node.aabb_z.y, node.aabb_z.z, node.aabb_z.w);

        // Unpack all at once using bit shifts
        vec4 minX = vec4(ax & 0xFFFFu) / 65535.0;
        vec4 maxX = vec4(ax >> 16) / 65535.0;

        vec4 minY = vec4(ay & 0xFFFFu) / 65535.0;
        vec4 maxY = vec4(ay >> 16) / 65535.0;

        vec4 minZ = vec4(az & 0xFFFFu) / 65535.0;
        vec4 maxZ = vec4(az >> 16) / 65535.0;

        // Scale and offset to scene space
        minX = scene_min.x + minX * scene_extent.x;
        maxX = scene_min.x + maxX * scene_extent.x;

        minY = scene_min.y + minY * scene_extent.y;
        maxY = scene_min.y + maxY * scene_extent.y;

        minZ = scene_min.z + minZ * scene_extent.z;
        maxZ = scene_min.z + maxZ * scene_extent.z;

        vec4 tNear, tFar;
        bvec4 mask = intersectAABB4(ray, minX, maxX, minY, maxY, minZ, maxZ, tNear, tFar);

        // Also cull against current closest hit
        mask = mask && lessThan(tNear, vec4(bestHit.t));


        // Convert to arrays for sorting
        float tArr[4];
        int   idxArr[4];
        bool  validArr[4];

        tArr[0] = tNear.x; idxArr[0] = 0; validArr[0] = mask.x;
        tArr[1] = tNear.y; idxArr[1] = 1; validArr[1] = mask.y;
        tArr[2] = tNear.z; idxArr[2] = 2; validArr[2] = mask.z;
        tArr[3] = tNear.w; idxArr[3] = 3; validArr[3] = mask.w;

        // Sort children by tNear
        sort4(tArr, idxArr, validArr);

        // Extract index (30 bits)
        uvec4 index = node.metadata & INDEX_MASK;

        // Extract type (top 2 bits)
        uvec4 primType  = (node.metadata >> TYPE_SHIFT) & TYPE_MASK;

        // Visit in order
        for (int k = 3; k >= 0; k--) {
            if (!validArr[k]) continue;
            uint i = idxArr[k];

            uint prim_type = primType[i];

            if (prim_type != 3) {
                uint prim_idx  = index[i];

                Hit prim_hit;

                if (prim_type == TRIANGLE) {
                    Triangle prim = triangles[prim_idx];
                    prim_hit = ray_triangle_collision(ray.origin, ray.direction, prim.v0, prim.v1, prim.v2, prim.normal);
                    if (prim_hit.exists && prim_hit.t > 0.0 && prim_hit.t < bestHit.t - 1e-6) {
                        bestHit = prim_hit;
                        bestHit.material_index = prim.material;
                    }
                } else if (prim_type == SPHERE) {
                    Sphere prim = spheres[prim_idx];
                    prim_hit = ray_sphere_collision(ray.origin, ray.direction, prim.center, prim.radius, prim.radius2);
                    if (prim_hit.exists && prim_hit.t > 0.0 && prim_hit.t < bestHit.t - 1e-6) {
                        bestHit = prim_hit;
                        bestHit.material_index = prim.material;
                    }
                } else if (prim_type == BOX) {
                    Box prim = boxes[prim_idx];
                    prim_hit = ray_box_collision(ray.origin, ray.direction, prim.p0, prim.p1);
                    if (prim_hit.exists && prim_hit.t > 0.0 && prim_hit.t < bestHit.t - 1e-6) {
                        bestHit = prim_hit;
                        bestHit.material_index = prim.material;
                    }
                }
            } else {
                stack[stackPtr++] = index[i];
            }
        }
    }

    return bestHit;
}

Hit calculate_ray_bvh(Ray ray) {
    Hit primitives_hit = traverse_bvh(ray);

    Hit plane_hit = calculate_planes(ray.origin, ray.direction, false, 10000);

    // Hit triangle_hit = calculate_triangles(ray.origin, ray.direction, false, 10000);
    
    //Hit box_hit = calculate_boxes(ray_origin, ray_direction, shadow, 10000);

    // Hit nearest_hit = find_nearest_hit(primitives_hit, plane_hit, plane_hit, plane_hit);

    // return nearest_hit;
    // return primitives_hit;

    if (primitives_hit.t < plane_hit.t) {
        return primitives_hit;
    } else {
        return plane_hit;
    }
}

vec3 sampleCosineWeightedHemisphere(inout uint random_state) {
    float u1 = random(random_state);
    float u2 = random(random_state);

    float r = sqrt(u2);
    float theta = 2.0 * 3.14159265 * u1;

    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0, 1.0 - u2)); // z is the cosine of the angle

    return vec3(x, y, z); // Returns a direction in tangent space
}

// GLSL
// Creates a coordinate system from a single vector (the normal)
mat3 createOrthonormalBasis(vec3 normal) {
    vec3 tangent;
    if (abs(normal.x) > abs(normal.y)) {
        tangent = vec3(normal.z, 0, -normal.x) / length(vec3(normal.z, 0, -normal.x));
    } else {
        tangent = vec3(0, -normal.z, normal.y) / length(vec3(0, -normal.z, normal.y));
    }
    vec3 bitangent = cross(normal, tangent);
    return mat3(tangent, bitangent, normal);
}

// Creates a local coordinate system from a normal vector N
void create_orthonormal_basis(vec3 N, out vec3 T, out vec3 B) {
    if (abs(N.x) > abs(N.y)) {
        T = vec3(-N.z, 0, N.x) / sqrt(N.x * N.x + N.z * N.z);
    } else {
        T = vec3(0, N.z, -N.y) / sqrt(N.y * N.y + N.z * N.z);
    }
    B = cross(N, T);
}

// Generates a sample microfacet normal in tangent space using GGX
// This is the heart of the rough reflection logic
vec3 importance_sample_ggx(float rand1, float rand2, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;

    float phi = 2.0 * 3.14159265 * rand1;
    float cos_theta = sqrt((1.0 - rand2) / (1.0 + (a2 - 1.0) * rand2));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    vec3 H_local;
    H_local.x = cos(phi) * sin_theta;
    H_local.y = sin(phi) * sin_theta;
    H_local.z = cos_theta;

    return H_local;
}

vec3 unpackRGB(uint packedValue) {
    // Masks
    uint maskR = 0x7FFu; // 11 bits
    uint maskG = 0x7FFu; // 11 bits
    uint maskB = 0x3FFu; // 10 bits

    // Extract
    uint r = (packedValue >> 21) & maskR;   // top 11 bits
    uint g = (packedValue >> 10) & maskG;   // next 11 bits
    uint b = packedValue & maskB;           // lowest 10 bits

    // Normalize to [0,1]
    return vec3(
        float(r) / 2047.0, // 0x7FF
        float(g) / 2047.0,
        float(b) / 1023.0  // 0x3FF
    );
}

// Russian Roulette Termination
bool russianRoulette(inout vec3 throughput, uint depth, inout uint rngState) {
    const uint minDepth = 3;   // don’t kill before this depth
    const uint maxDepth = 64;  // hard cutoff for safety

    if (depth < minDepth) return true;
    if (depth >= maxDepth) return false;

    // Luminance-based survival probability
    float lum = dot(throughput, vec3(0.2126, 0.7152, 0.0722));
    float p = clamp(lum, 0.05, 0.95);  // keep inside sane range

    if (random(rngState) > p) {
        return false; // kill path
    }

    throughput /= p; // unbiased scaling
    return true;     // survive
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
    vec2 jitter = vec2(random(rng_state) - 0.5, random(rng_state) - 0.5) / vec2(dims);


    // color vector
    vec3 color = vec3(1.0, 1.0, 1.0);


    Ray ray;
    
    // store the eye position as the origin of the ray
    ray.origin = eye;

    // calculate the direction of the ray based on the camera matrix
    ray.direction = camera_ray_direction(position.xy + jitter, inverse_view_projection);

    ray.inv_direction = 1 / ray.direction;

    ray.ior = 1.0;

    // declare the result of the ray hit
    Hit hit;

    // prepare the amount of light captured by the ray
    vec3 light = vec3(0);
    bool refracted = false;
    bool transmitted = false;
    bool reflected = false;
    bool entering = false;
    
    // iterate through the number of bounces
    for (uint i = 0; i < uint(bounces); i++) {
        // do the ray intersection calculationù
        hit = calculate_ray_bvh(ray);

        // if the ray didn't hit anything, break out of the loop
        if (!hit.exists) {
            // light += vec3(0.0, 0.3, 1.0) * color;
            break;
        }

        mat3 tangentSpace = createOrthonormalBasis(hit.normal);

        // extract the material from the hit
        Material material = materials[hit.material_index];

        // extract the color from the material
        vec3 material_color = srgb_to_rgb(unpackRGB(material.color));

        // extract the physics properties from the material
        vec4 properties = unpackFloat4x8(material.properties);

        // extract the specific properties
        float smoothness = properties.x;
        float metallic = properties.y;
        float transmission = properties.z;
        float ior = properties.w * 3.0;

        // extract the emission
        vec4 formatted_emission = unpackFloat4x8(material.emission);
        uint emission_strength = (material.emission >> 24u) & 0xFFu;
        vec3 emission = srgb_to_rgb(formatted_emission.xyz);
        
        // get the emitted light by the color multiplied by the strength
        vec3 emitted_light = emission * emission_strength;

        // add the emitted light multiplied by the color to get the total light
        light += emitted_light * color;

        // if we reached the last bounce, we break early
        // if (i == uint(bounces) + 1) {
        //   break;
        // }

        // 1. Determine surface properties by mixing between dielectric and metallic
        vec3 dielectric_specular_tint = vec3(1.0); // Dielectrics have white reflections
        vec3 metallic_specular_tint = material_color; // Metals have colored reflections
        vec3 surface_specular_tint = mix(dielectric_specular_tint, metallic_specular_tint, metallic);

        // 2. Calculate Fresnel reflectance (how much light reflects off the surface)
        // The F0 value (reflectance at normal incidence) also blends.
        float f0_dielectric = pow((1.0 - ior) / (1.0 + ior), 2.0);
        float f0 = mix(f0_dielectric, 1.0, metallic); // F0 for metals is effectively 1.0

        // Using Schlick's approximation for angle-dependent reflectance
        float cos_theta = min(dot(-ray.direction, hit.normal), 1.0);
        float reflectance = f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);

        // 3. Make the main decision: Reflect or Transmit/Scatter?
        if (random(rng_state) < reflectance) {
            reflected = true;
            vec3 reflected_dir;

            if (smoothness == 1.0) {
                // Perfect mirror reflection
                ray.direction = reflect(ray.direction, hit.normal);
            } else {
                // Rough reflection
                // 1. Get two random numbers
                float rand1 = random(rng_state);
                float rand2 = random(rng_state);

                // 2. Sample a microfacet normal in local (tangent) space
                vec3 H_local = importance_sample_ggx(rand1, rand2, 1 - smoothness);

                // 3. Create the local coordinate system around the geometric normal
                vec3 T, B;
                create_orthonormal_basis(hit.normal, T, B);

                // 4. Transform the microfacet normal to world space
                vec3 H = normalize(T * H_local.x + B * H_local.y + hit.normal * H_local.z);

                // 5. Calculate the final reflection direction
                ray.direction = reflect(ray.direction, H);
            }

            ray.origin = hit.pos + ray.direction * 0.0001;
            color *= surface_specular_tint;
            // --- SURFACE REFLECTION (Specular) ---
            // This branch handles both metallic and dielectric reflections.
            // ray.direction = reflect(ray.direction, hit.normal); // Add roughness perturbation here for non-mirrors
            // ray.origin = hit.pos + ray.direction * 0.0001;
            // color *= surface_specular_tint;

        } else {
            refracted = true;
            // --- NOT REFLECTED: The light either transmits or scatters underneath ---

            // 4. Decide between Transmission (glass) and Diffuse (plastic)
            // This decision only happens for the portion of light that isn't reflected.
            if (random(rng_state) < transmission) {
                transmitted = true;
                // --- TRANSMISSION / REFRACTION (Glass-like behavior) ---
                // This is your previous glass logic for refraction
                entering = dot(ray.direction, hit.normal) < 0.0;
                vec3 normal = entering ? hit.normal : -hit.normal;
                float n1 = ray.ior; // Use the ray's current IOR
                float n2 = entering ? ior : 1.0; // Target IOR is object's or air's

                float rand1 = random(rng_state);
                float rand2 = random(rng_state);

                vec3 H_local = importance_sample_ggx(rand1, rand2, 1 - smoothness);
                vec3 T, B;
                create_orthonormal_basis(hit.normal, T, B);
                vec3 H = normalize(T * H_local.x + B * H_local.y + hit.normal * H_local.z);

                vec3 refracted_direction = refract(ray.direction, H, n1 / n2);
                
                // vec3 refracted_direction = refract(ray.direction, normal, n1 / n2);
                // Handle Total Internal Reflection if refract returns vec3(0.0)

                // Check for Total Internal Reflection

                if (dot(refracted_direction, refracted_direction) == 0.0) {
                    ray.direction = reflect(ray.direction, normal);
                } else {
                    ray.direction = refracted_direction;
                    ray.ior = n2;
                }
                
                ray.origin = hit.pos + ray.direction * 0.0001;
                // The tint is applied by the baseColor. For colored glass, you'd apply Beer's Law.
                // color *= material_color;

            } else {
                // --- SUBSURFACE SCATTERING (Diffuse behavior) ---
                // This logic will be skipped entirely if transmission = 1.0
                // And will be the only option if transmission = 0.0 (for dielectrics)
                // ... your existing diffuse logic (e.g., sample a random direction in the hemisphere) ...
                // pick a random direction from the normal to calculate the next ray for diffuse lighting 
                // The improved way to get a diffuse direction
                vec3 diffuse_direction = tangentSpace * sampleCosineWeightedHemisphere(rng_state);
                // calculate the reflection ray from the surface to get the specular lighting
                vec3 specular_direction = normalize(ray.direction - 2 * hit.normal * dot(ray.direction, hit.normal));

                ray.origin = hit.pos + hit.normal * 0.0001;

                // calculate the direction of the bounce by mixing the specular and diffuse directions with the smoothness and albedo chance
                ray.direction = mix(diffuse_direction, specular_direction, smoothness);

                // if the obtained direction is opposite to the normal, invert it
                if (dot(ray.direction, hit.normal) < 0) {
                    ray.direction = -ray.direction;
                }

                // calculate the light strength based on the angle between the normal and the bounce direction
                float light_strength = dot(hit.normal, ray.direction);

                color *= material_color * (1.0 - metallic); // Metals absorb diffuse light
            }
        }

        ray.inv_direction = 1 / ray.direction;

        if (!russianRoulette(color, i, rng_state)) {
            break;
        }
        // ray.ior = ior;

        // get the color for the next hit
        // color *= material_color;
    }
    

    // store the pixel in the texture
    imageStore(img_output, texel_coord, vec4(rgb_to_srgb(light), 1.0));

    // hit = ray_triangle_collision(ray.origin, ray.direction, triangles[0].v0, triangles[0].v1, triangles[0].v2, triangles[0].normal);
    // hit = ray_sphere_collision(ray.origin, ray.direction, spheres[1].center, spheres[1].radius, spheres[1].radius2);

    // if (hit.exists) {
    //     imageStore(img_output, texel_coord, vec4(1.0));
    // } else {
    //     imageStore(img_output, texel_coord, vec4(0.0, 0.0, 0.0, 1.0));
    // }

    // vec4 formatted_emission = unpackFloat4x8(materials[0].emission);
    // // uint emission_strength = (material.emission >> 24u) & 0xFFu;

    // imageStore(img_output, texel_coord, vec4(formatted_emission.xyz, 1.0));
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

uint xorshift(inout uint state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
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
    return wang_hash(state) * INV_UINT_MAX_PLUS_ONE;
}


float map(float x, float in_min, float in_max, float out_min, float out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

Hit ray_plane_collision(vec3 ray_origin, vec3 ray_direction, vec3 position, vec3 normal) {
    Hit hit;
    hit.exists = false;

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

Hit calculate_planes(vec3 ray_origin, vec3 ray_direction, bool shadow, float max_t) {
    Hit nearest_hit;
    nearest_hit.exists = false;
    nearest_hit.t = 9999999;

    for (uint i = 0; i <= planes.length(); i++) {
        vec3 position = vec3(planes[i].center.x, planes[i].center.y, planes[i].center.z);
        vec3 normal = vec3(planes[i].normal.x, planes[i].normal.y, planes[i].normal.z);

        Hit hit = ray_plane_collision(ray_origin, ray_direction, planes[i].center, planes[i].normal);
        if (hit.exists && hit.t < nearest_hit.t) {
            hit.material_index = uint(planes[i].material);
            nearest_hit = hit;
        }
    }

    return(nearest_hit);
}


Hit ray_sphere_collision(vec3 ray_origin, vec3 ray_direction, vec3 sphere_center, float sphere_radius, float sphere_radius2) {
        Hit hit;
        hit.exists = false;

        float t0, t1; // solutions for t if the ray intersects

        // geometric solution
        vec3 L = sphere_center - ray_origin;
        float tca = dot(L, ray_direction);
        // if (tca < 0) return hit;
        float d2 = dot(L, L) - tca * tca;
        if (d2 > sphere_radius2) return(hit);
        float thc = sqrt(sphere_radius2 - d2);
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

        if (dot(ray_direction, hit.normal) > 0.0) {
            hit.normal = -hit.normal; // flip if inside
        }
        
        return(hit);
}

Hit calculate_spheres(vec3 ray_origin, vec3 ray_direction, bool shadow, float max_t) {
    Hit nearest_hit;
    nearest_hit.exists = false;
    nearest_hit.t = 999999;

    for (uint i = 0; i <= spheres.length(); i++) {
        vec3 center = spheres[i].center;
        float radius = spheres[i].radius;
        float radius2 = spheres[i].radius2;

        Hit hit = ray_sphere_collision(ray_origin, ray_direction, center, radius, radius2);
        hit.material_index = uint(spheres[i].material);

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

Hit ray_triangle_collision(vec3 ray_origin, vec3 ray_direction, vec3 v0, vec3 v1, vec3 v2, vec3 normal) {
    Hit result;
    result.exists = false;
    result.normal = normal;

    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;

    // Begin calculating determinant
    vec3 pvec = cross(ray_direction, edge2);
    float det = dot(edge1, pvec);

    // Cull backfaces if desired: if(det < EPSILON) return result;
    if (abs(det) < 1e-6) return result; // Ray parallel to triangle

    float invDet = 1.0 / det;

    // Distance from v0 to ray origin
    vec3 tvec = ray_origin - v0;

    // Calculate u parameter
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0 || u > 1.0) return result;

    // Prepare to test v parameter
    vec3 qvec = cross(tvec, edge1);

    float v = dot(ray_direction, qvec) * invDet;
    if (v < 0.0 || u + v > 1.0) return result;

    // At this stage we can compute t
    float t = dot(edge2, qvec) * invDet;
    if (t <= 0.0) return result; // No intersection in front of ray

    // Fill result
    result.exists = true;
    result.t = t;
    result.pos = ray_origin + t * ray_direction;

    if (dot(ray_direction, normal) > 0) {
        result.normal = -result.normal;
    }

    return result;
}

Hit ray_box_collision(vec3 ray_origin, vec3 ray_direction, vec3 b0, vec3 b1) {
    Hit hit;
    hit.exists = false;

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

Hit calculate_boxes(vec3 ray_origin, vec3 ray_direction, bool shadow, float max_t) {
    Hit nearest_hit;
    nearest_hit.exists = false;
    nearest_hit.t = 9999999;

    for (uint i = 0; i <= boxes.length(); i ++) {
        Box box = boxes[i];

        Hit hit = ray_box_collision(ray_origin, ray_direction, box.p0, box.p1);
        hit.material_index = box.material;

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


Hit calculate_triangles(vec3 ray_origin, vec3 ray_direction, bool shadow, float max_t) {
    Hit nearest_hit;
    nearest_hit.exists = false;
    nearest_hit.t = 9999999;

    for (uint i = 0; i <= triangles.length(); i++) {
        Hit hit = ray_triangle_collision(ray_origin, ray_direction, triangles[i].v0, triangles[i].v1, triangles[i].v2, triangles[i].normal);
        hit.material_index = uint(triangles[i].material);

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

Hit find_nearest_hit(Hit h0, Hit h1, Hit h2, Hit h3) {
    Hit nearest_hit;
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

Hit calculate_ray(vec3 ray_origin, vec3 ray_direction, bool shadow) {

    //Hit triangle_hit = calculate_triangles(ray_origin, ray_direction, shadow, 10000);

    Hit sphere_hit = calculate_spheres(ray_origin, ray_direction, shadow, 10000);

    Hit plane_hit = calculate_planes(ray_origin, ray_direction, shadow, 10000);
    
    //Hit box_hit = calculate_boxes(ray_origin, ray_direction, shadow, 10000);

    Hit nearest_hit = find_nearest_hit(sphere_hit, plane_hit, sphere_hit, plane_hit);


    return(nearest_hit);
    //return(sphere_hit);
}

/**
 * Unpacks a 32-bit unsigned integer into four 8-bit components (a vec4).
 * The integer components are then normalized to the [0.0, 1.0] float range.
 * This is the reverse of packing four uint8 values into a uint32.
 * Assumes Little Endian packing order (R at the lowest bits).
 */
vec4 unpackFloat4x8(uint packedValue) {
    uint r = (packedValue) & 0xFFu;        // Mask the lowest 8 bits for Red
    uint g = (packedValue >> 8u) & 0xFFu;  // Shift right by 8, then mask for Green
    uint b = (packedValue >> 16u) & 0xFFu; // Shift right by 16, then mask for Blue
    uint a = (packedValue >> 24u) & 0xFFu; // Shift right by 24, then mask for Alpha

    // Convert uints [0, 255] to floats [0.0, 1.0] and return as a vec4
    return vec4(r, g, b, a) / 255.0;
}

/**
 * Unpacks a 32-bit unsigned integer into two 16-bit components.
 *
 * @param packedValue The uint32 to unpack.
 * @return A uvec2 where .x is the lower 16 bits and .y is the upper 16 bits.
 */
uvec2 unpackUint2x16(uint packedValue) {
    // For the first value, use a bitwise AND with a mask to isolate the lower 16 bits.
    // 0xFFFFu is hexadecimal for 65535, which is 16 ones in binary.
    uint val1 = packedValue & 0xFFFFu;

    // For the second value, shift the bits 16 positions to the right.
    // This discards the lower 16 bits and moves the upper 16 bits into place.
    uint val2 = packedValue >> 16u;

    return uvec2(val1, val2);
}
