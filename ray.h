#ifndef RAY_H
#define RAY_H

#include <math.h>

#include "common.h"
#include "point3.h"
#include "vec3.h"

typedef struct {
    point3 origin;
    vec3 direction;
} ray;

// Constructor
ray ray_new(const point3 origin, const vec3 direction) {
    ray r = {origin, direction};
    return r;
}

// Function to get the point at a given time t along the ray
point3 ray_at(const ray r, my_decimal t) {
    return sum(r.origin, scale(r.direction, t));
}

void ray_print(ray r) {
    fprintf(stderr, "Ray");
    printf("  direction=");
    vec3_print(r.direction);
    printf("\n  origin=");
    point3_print(r.origin); 
}

#endif
