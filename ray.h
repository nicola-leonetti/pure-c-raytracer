#ifndef RAY_H
#define RAY_H

#include <math.h>

#include "common.h"
#include "point3.h"
#include "vec3.h"

typedef struct {
    t_point3 origin;
    t_vec3 direction;
} t_ray;

// Constructor
t_ray ray_new(const t_point3 origin, const t_vec3 direction) {
    t_ray r = {origin, direction};
    return r;
}

// Function to get the point at a given time t along the ray
t_point3 ray_at(const t_ray r, my_decimal t) {
    return sum(r.origin, scale(r.direction, t));
}

void ray_print(t_ray r) {
    fprintf(stderr, "Ray");
    printf("  direction=");
    vec3_print(r.direction);
    printf("\n  origin=");
    point3_print(r.origin); 
}

#endif
