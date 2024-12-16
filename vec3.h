#ifndef VEC3_H
#define VEC3_H

#include <math.h>
#include <stdio.h>

#include "common.h"

typedef struct {
    my_decimal x;
    my_decimal y;
    my_decimal z;
} vec3;

// Constructor
vec3 vec3_new(my_decimal e0, my_decimal e1, my_decimal e2) {
    vec3 v = {e0, e1, e2};
    return v;
}

vec3 vec3_sum(vec3 u, const vec3 v) {
    return vec3_new((u.x)+(v.x), (u.y)+(v.y), (u.z)+(v.z));
}
vec3 vec3_subtract(vec3 u, const vec3 v) {
    return vec3_new((u.x)-(v.x), (u.y)-(v.y), (u.z)-(v.z));
}

my_decimal vec3_length_squared(const vec3 v) {
    return (v.x)*(v.x) + (v.y)*(v.y) + (v.z)*(v.z);
}
my_decimal vec3_length(const vec3 v) {
    return sqrt(vec3_length_squared(v));
}

vec3 vec3_scale(vec3 v, my_decimal t) {
    return vec3_new(t*(v.x), t*(v.y), t*(v.z));
}
vec3 vec3_divide(vec3 v, my_decimal t) {
    return vec3_scale(v, 1.0/t);
}
vec3 vec3_negate(const vec3 v) {
    return vec3_scale(v, -1);
}
// Returns a vector with the same direction, but length 1
vec3 vec3_unit(vec3 v) {
    return vec3_divide(v, vec3_length(v));
}

vec3 vec3_multiply(const vec3 u, const vec3 v) {
    return vec3_new((u.x)*(v.x), (u.y)*(v.y), (u.z)*(v.z));
}
my_decimal vec3_dot(const vec3 u, const vec3 v) {
    return (u.x)*(v.x) + (u.y)*(v.y) + (u.z)*(v.z);
}
vec3 vec3_cross(const vec3 u, const vec3 v) {
    return vec3_new(
        (u.y)*(v.z) - (u.z)*(v.y), 
        (u.z)*(v.x) - (u.x)*(v.z),
        (u.x)*(v.y) - (u.y)*(v.x)
    );
}

void vec3_print(const vec3 v) {
    printf("Vector(%f, %f, %f)\n", v.x, v.y, v.z);
}

#endif
