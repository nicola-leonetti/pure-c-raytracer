#ifndef SPHERE_H
#define SPHERE_H

#include "common.h"
#include "hittable.h"
#include "point3.h"
#include "ray.h"

typedef struct {
    point3 center;
    my_decimal radius;
} sphere;

sphere sphere_new(point3 center, my_decimal radius) {
    sphere s = {center, radius};
    return s;
}

// Returns hit point and normal vector of a ray with a sphere.
// For the details of how this calculation is performed, refer to the sources 
// in README.md
hit_result sphere_hit(ray r, sphere s, my_decimal t_min, my_decimal t_max) {
    hit_result result;

    vec3 oc = vec3_subtract(s.center, r.origin);

    my_decimal a = vec3_length_squared(r.direction);
    my_decimal h = vec3_dot(r.direction, oc);
    my_decimal c = vec3_length_squared(oc) - (s.radius)*(s.radius);
    my_decimal discriminant = h*h - a*c;

    // discriminant < 0 -> no real solutions -> no intersections
    if (discriminant < 0) {
        result.did_hit = false;
        return result;
    }

    // Check for each of the intersections if it lies in the acceptable range
    my_decimal sqrt_discriminant = sqrt(discriminant);
    my_decimal root = (h - sqrt(discriminant)) / a;

    // First root not in the acceptable range (t_min, t_max)
    if (root <= t_min || root >= t_max) {
        root = (h + sqrt(discriminant)) / a;
        // None of the two roots is in the acceptable range (t_min, t_max)
        if (root <= t_min || root >= t_max) {
            result.did_hit = false;
            return result;
        }
    }

    // Calculate intersection and return result
    result.did_hit = true;
    result.t = root;
    result.p = ray_at(r, result.t);
    // Calculating the normal vector with this formula, it always points
    // outwards
    result.normal = vec3_divide(vec3_subtract(result.p, s.center), s.radius);

    // In order to calculate wether we hit an inside or ourside face, we can 
    // compute the dot product with the (OUTSIDE-POINTING!!!) normal
    result.front_face = (vec3_dot(r.direction, result.normal) < 0);

    // TODO Se non funziona, devo invertire il segno della normale quando il
    // raggio colpisce l'oggetto da dentro
    // if (!result.front_face) {
    //     result.normal = vec3_scale(result.normal, -1);
    // }

    return result;
}

#endif 