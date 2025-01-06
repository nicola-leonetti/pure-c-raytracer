#ifndef VEC3_H
#define VEC3_H

#include <math.h>
#include <stdio.h>

#include "common.h"

typedef struct { float x, y, z; } t_vec3;

// Constructor
__host__ __device__ t_vec3 vec3_new(
    float e0, 
    float e1, 
    float e2
) {
    return (t_vec3) {e0, e1, e2};
}

__host__ __device__ t_vec3 sum(t_vec3 u, t_vec3 v) {
    return vec3_new((u.x)+(v.x), (u.y)+(v.y), (u.z)+(v.z));
}

__host__ __device__ t_vec3 subtract(t_vec3 u, t_vec3 v) {
    return vec3_new((u.x)-(v.x), (u.y)-(v.y), (u.z)-(v.z));
}
    
__host__ __device__ float squared_length(t_vec3 v) {
    return (v.x)*(v.x) + (v.y)*(v.y) + (v.z)*(v.z);
}

__host__ __device__ float length(t_vec3 v) {
    return sqrt(squared_length(v));
}

__host__ __device__ t_vec3 scale(t_vec3 v, float t) {
    return vec3_new(t*(v.x), t*(v.y), t*(v.z));
}

__host__ __device__ t_vec3 divide(t_vec3 v, float t) {
    return scale(v, 1.0F/t);
}

__host__ __device__ t_vec3 negate(const t_vec3 v) {
    return scale(v, -1.0F);
}

// Returns vector with same direction and unitary length
__host__ __device__ t_vec3 vec3_unit(t_vec3 v) {
    return divide(v, length(v));
}

// Element-wise multiplication
__host__ __device__ t_vec3 mul(t_vec3 u, t_vec3 v) {
    return vec3_new((u.x)*(v.x), (u.y)*(v.y), (u.z)*(v.z));
}

// Dot product
__host__ __device__ float dot(t_vec3 u, t_vec3 v) {
    return (u.x)*(v.x) + (u.y)*(v.y) + (u.z)*(v.z);
}

// Cross product
__host__ __device__ t_vec3 cross(t_vec3 u, t_vec3 v) {
    return (t_vec3) {
        (u.y)*(v.z) - (u.z)*(v.y), 
        (u.z)*(v.x) - (u.x)*(v.z),
        (u.x)*(v.y) - (u.y)*(v.x)
    };
}

__host__ void vec3_print(t_vec3 v) {
    fprintf(stderr, "Vector(%f, %f, %f)\n", v.x, v.y, v.z);
}

// Random vector with each component in [0, 1)
__host__ t_vec3 h_vec3_random() {
    return (t_vec3) {
        h_random_float(), 
        h_random_float(), 
        h_random_float()
    };
}
__device__ t_vec3 d_vec3_random(curandState *state) {
    return (t_vec3) {
        d_random_float(state), 
        d_random_float(state), 
        d_random_float(state)
    };
}

// TODO Optimize
// Random unit vector
__host__ t_vec3 h_vec3_random_unit() {
    while (true) {
        t_vec3 point = h_vec3_random();
        float length_squared = squared_length(point);
        if (float_UNDERFLOW_LIMIT < length_squared && \
                                         length_squared <= 1.0F) {
            return divide(point, sqrt(length_squared));
        }
    }
}
__device__ t_vec3 d_vec3_random_unit(curandState *state) {
    while (true) {
        t_vec3 point = d_vec3_random(state);
        float length_squared = squared_length(point);
        if (float_UNDERFLOW_LIMIT < length_squared && \
                                         length_squared <= 1.0F) {
            return divide(point, sqrt(length_squared));
        }
    }
}

// Random unit vector in a unit disk
__host__ t_vec3 h_random_in_unit_disk() {
    while (true) {
        t_vec3 p = vec3_new(
            h_random_float_in(-1.0F, 1.0F), 
            h_random_float_in(-1.0F, 1.0F), 
            0
        );
        if (squared_length(p) < 1.0F) {
            return p;
        }
    }
}
__device__ t_vec3 d_random_in_unit_disk(curandState *state) {
    while (true) {
        t_vec3 p = vec3_new(
            d_random_float_in(-1.0F, 1.0F, state), 
            d_random_float_in(-1.0F, 1.0F, state), 
            0);
        if (squared_length(p) < 1.0F) {
            return p;
        }
    }
}

#define NEAR_ZERO(v) \
    (fabs(v.x) < NEAR_ZERO_TRESHOLD) && \
    (fabs(v.y) < NEAR_ZERO_TRESHOLD) && \
    (fabs(v.z) < NEAR_ZERO_TRESHOLD)

// Calculate reflected ray over a surface with given normal
#define REFLECT(v, n) \
    subtract(v, scale(n, 2.0F*dot(v, n)))

// Calculate refracted ray over a surface with given normal and a given ratio 
// of refractive indexes
__host__ __device__ t_vec3 refract(t_vec3 v, t_vec3 n, float ratio) {
    float cos_theta = fmin(dot(negate(v), n), 1.0F);
    t_vec3 out_perpendicular = scale(sum(v, scale(n, cos_theta)), ratio);
    t_vec3 out_parallel = scale(
        n, -sqrt(fabs(1 - squared_length(out_perpendicular))));
    return(sum(out_perpendicular, out_parallel));
}

#endif
