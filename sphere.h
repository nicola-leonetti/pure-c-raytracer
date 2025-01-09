#ifndef SPHERE_H
#define SPHERE_H

#include "color.h"
#include "common.h"
#include "hittable.h"
#include "material.h"
#include "point3.h"
#include "ray.h"

typedef struct {
    t_point3 center;
    float radius;
    t_material material;
} t_sphere;

__host__ __device__ t_sphere sphere_new(
    t_point3 center, 
    float radius, 
    t_material material
) {
    return (t_sphere) { center, radius, material };
}

// Returns hit point and normal vector of a ray with a sphere.
// For the details of how this calculation is performed, refer to the sources 
// in README.md
__host__ void h_sphere_hit(
    t_hit_result *result, 
    t_ray *r, 
    t_sphere s, 
    float t_min, 
    float t_max
) {

    t_vec3 oc = subtract(s.center, r->origin);

    float a = squared_length(r->direction);
    float h = dot(r->direction, oc);
    float c = squared_length(oc) - (s.radius)*(s.radius);
    float discriminant = h*h - a*c;

    // discriminant < 0 -> no real solutions -> no intersections
    if (discriminant < 0) {
        result->did_hit = false;
        return;
    }

    // Check for each of the intersections if it lies in the acceptable range
    float sqrt_discriminant = sqrt(discriminant);
    float root = (h - sqrt(discriminant)) / a;

    // First root not in the acceptable range (t_min, t_max)
    if (root <= t_min || root >= t_max) {
        root = (h + sqrt(discriminant)) / a;
        // None of the two roots is in the acceptable range (t_min, t_max)
        if (root <= t_min || root >= t_max) {
            result->did_hit = false;
            return;
        }
    }

    // Calculate intersection
    result->did_hit = true;
    result->t = root;
    ray_at(&(result->p), *r, result->t);

    // Calculating the normal vector with this formula, it always points
    // outwards
    result->normal = divide(subtract(result->p, s.center), s.radius);

    // In order to calculate wether we hit an inside or ourside face, we can 
    // compute the dot product with the (OUTSIDE-POINTING!!!) normal
    result->front_face = (dot(r->direction, result->normal) < 0.0F);

    // TOTO switch to just memorizing the sphere object
    result->albedo = s.material.albedo;
    result->surface_material = s.material.type;
    result->fuzz = s.material.fuzz;
    result->refraction_index = s.material.refraction_index;

    // If the ray hits an object from inside (like in dielectrics, I need to 
    // invert the direction of the normal
    if (!result->front_face) {
        result->normal = scale(result->normal, -1.0F);
    }
}

__device__ void d_sphere_hit(
    t_hit_result *result, 
    t_ray *r, 
    t_sphere s, 
    float t_min, 
    float t_max
) {

    t_vec3 oc = subtract(s.center, r->origin);

    float a = squared_length(r->direction);
    float h = dot(r->direction, oc);
    float c = squared_length(oc) - (s.radius)*(s.radius);
    float discriminant = h*h - a*c;

    // discriminant < 0 -> no real solutions -> no intersections
    if (discriminant < 0) {
        result->did_hit = false;
        return;
    }

    // Check for each of the intersections if it lies in the acceptable range
    float sqrt_discriminant = sqrt(discriminant);
    float root = (h - sqrt_discriminant) / a;

    // First root not in the acceptable range (t_min, t_max)
    if (root <= t_min || root >= t_max) {
        root = (h + sqrt_discriminant) / a;
        // None of the two roots is in the acceptable range (t_min, t_max)
        if (root <= t_min || root >= t_max) {
            result->did_hit = false;
            return;
        }
    }

    // Calculate intersection
    result->did_hit = true;
    result->t = root;
    ray_at(&(result->p), *r, result->t);

    // Calculating the normal vector with this formula, it always points
    // outwards
    result->normal = divide(subtract(result->p, s.center), s.radius);

    // In order to calculate wether we hit an inside or ourside face, we can 
    // compute the dot product with the (OUTSIDE-POINTING!!!) normal
    result->front_face = (dot(r->direction, result->normal) < 0.0F);

    // TOTO switch to just memorizing the sphere object
    result->albedo = s.material.albedo;
    result->surface_material = s.material.type;
    result->fuzz = s.material.fuzz;
    result->refraction_index = s.material.refraction_index;

    // If the ray hits an object from inside (like in dielectrics, I need to 
    // invert the direction of the normal
    if (!result->front_face) {
        result->normal = scale(result->normal, -1.0F);
    }
}

#endif 