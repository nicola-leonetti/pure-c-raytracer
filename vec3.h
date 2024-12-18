#ifndef VEC3_H
#define VEC3_H

#include <math.h>
#include <stdio.h>

#include "common.h"

typedef struct { my_decimal x, y, z; } vec3;

// Constructor
vec3 vec3_new(my_decimal e0, my_decimal e1, my_decimal e2) {
    return (vec3) {e0, e1, e2};
}

vec3 sum(vec3 u, vec3 v) {
    return vec3_new((u.x)+(v.x), (u.y)+(v.y), (u.z)+(v.z));
}

vec3 subtract(vec3 u, vec3 v) {
    return vec3_new((u.x)-(v.x), (u.y)-(v.y), (u.z)-(v.z));
}
    
my_decimal squared_length(vec3 v) {
    return (v.x)*(v.x) + (v.y)*(v.y) + (v.z)*(v.z);
}

my_decimal length(vec3 v) {
    return sqrt(squared_length(v));
}

vec3 scale(vec3 v, my_decimal t) {
    return vec3_new(t*(v.x), t*(v.y), t*(v.z));
}

vec3 divide(vec3 v, my_decimal t) {
    return scale(v, 1.0/t);
}

vec3 negate(const vec3 v) {
    return scale(v, -1);
}

// Returns vector with same direction and unitary length
vec3 vec3_unit(vec3 v) {
    return divide(v, length(v));
}

// Element-wise multiplication
vec3 mul(vec3 u, vec3 v) {
    return vec3_new((u.x)*(v.x), (u.y)*(v.y), (u.z)*(v.z));
}

// Dot product
my_decimal dot(vec3 u, vec3 v) {
    return (u.x)*(v.x) + (u.y)*(v.y) + (u.z)*(v.z);
}

// Cross product
vec3 cross(vec3 u, vec3 v) {
    return (vec3) {
        (u.y)*(v.z) - (u.z)*(v.y), 
        (u.z)*(v.x) - (u.x)*(v.z),
        (u.x)*(v.y) - (u.y)*(v.x)
    };
}

void vec3_print(vec3 v) {
    fprintf(stderr, "Vector(%f, %f, %f)\n", v.x, v.y, v.z);
}

// Random vector with each component in [0, 1)
vec3 vec3_random() {
    return (vec3) {
        random_my_decimal(), 
        random_my_decimal(), 
        random_my_decimal()
    };
}

// Random vector with each component in [min, max)
vec3 vec3_random_in(my_decimal min, my_decimal max) {
    return (vec3) {
        random_my_decimal_in(min, max), 
        random_my_decimal_in(min, max), 
        random_my_decimal_in(min, max) 
    };
}

// TODO Optimize
// Random unit vector
vec3 vec3_random_unit() {
    while (true) {
        vec3 point = vec3_random();
        my_decimal length_squared = squared_length(point);
        if (MY_DECIMAL_UNDERFLOW_LIMIT < length_squared && length_squared <= 1) {
            return divide(point, sqrt(length_squared));
        }
    }
}

// Random unit vector on a surface with given surface normal
vec3 vec3_random_on_hemisphere(vec3 normal) {
    vec3 v = vec3_random_unit();
    return (dot(v, normal) > 0) ? v : scale(v, -1.0); 
}

#endif
