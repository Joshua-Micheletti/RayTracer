#version 330 core

layout (location=0) in vec3 aPos;

out vec2 position;
//out float hit;

//out vec3 debug;

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

/*
void main() {
    position = vec2(aPos.xy);

    vec3 v0 = vec3(-0.5, -0.5, 0.0);
    vec3 v1 = vec3(0.5, -0.5, 0.0);
    vec3 v2 = vec3(0, 0.5, 0.0);

    vec3 edge01 = v1 - v0;
    vec3 edge02 = v2 - v0;

    vec3 normal = normalize(cross(edge01, edge02));

    //debug = normal;

    vec3 camera = vec3(0.0, 0.0, -1.0);

    vec3 pixel = vec3(aPos.x, aPos.y, 0.0);

    vec3 origin = camera;
    vec3 direction = normalize(pixel - camera);

    //debug = direction;

    float dist = -dot(normal, v0);

    //debug = vec3(dist);

    float parallelism = dot(normal, direction);

    //debug = vec3(parallelism);

    /*if (parallelism == 0.0) {
        hit = -1.0;
    }*/
    /*
    //else {
        float t = -(dot(normal, origin) + dist) / parallelism;

        //if (t <= 0.0) {
        //    hit = -1.0;
        //}
        
        //else {
            vec3 p_hit = origin + (t * direction);

            //debug = p_hit;

            if (inside_outside_test(v0, v1, v2, p_hit, normal) == 1.0) {
                hit = 1.0;
                debug = vec3(hit);
            }
            else {
                hit = -1.0;
                debug = vec3(hit);
            }

    debug = p_hit - v0;

            //debug = vec3(hit);
        //}
        //hit = 1;
    //}

    gl_Position = vec4(aPos, 1.0);
}*/

void main() {
    position = vec2(aPos.xy);
    gl_Position = vec4(aPos, 1.0);
}