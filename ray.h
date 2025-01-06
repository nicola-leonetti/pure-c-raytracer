#ifndef RAY_H
#define RAY_H

#include <math.h>

#include "common.h"
#include "hittable.h"
#include "material.h"
#include "point3.h"
#include "vec3.h"

typedef struct {
    t_point3 origin;
    t_vec3 direction;
} t_ray;

// Constructor
__host__ __device__ t_ray ray_new(
    const t_point3 origin, 
    const t_vec3 direction
) {
    t_ray r = {origin, direction};
    return r;
}

// Function to get the point at a given time t along the ray
__host__ __device__ void ray_at(
    t_point3 *point,
    const t_ray r, 
    float t
) {
    *point = sum(r.origin, scale(r.direction, t));
}

__host__ void ray_print(t_ray r) {
    fprintf(stderr, "Ray");
    fprintf(stderr, "\n  direction=");
    vec3_print(r.direction);
    fprintf(stderr, "  origin=");
    h_point3_print(r.origin); 
}

// Scatters the incident ray based on the material hit
__host__ void h_scatter(
    t_hit_result *hit_result,
    t_ray *ray_in, 
    t_color *attenuation, 
    t_ray *ray_scattered
) {

    t_vec3 scatter_direction;
    float refraction_index, cos_theta, sin_theta;
    bool cannot_refract;
    switch (hit_result->surface_material) {

        case LAMBERTIAN:
            scatter_direction = sum(hit_result->normal, h_vec3_random_unit());
            scatter_direction = NEAR_ZERO(scatter_direction) ?
                                    hit_result->normal : scatter_direction;
            *attenuation = hit_result->albedo;
            break;

        case METAL:
            scatter_direction = REFLECT(ray_in->direction, hit_result->normal);
            scatter_direction = sum(
                scatter_direction, 
                scale(h_vec3_random_unit(), 
                hit_result->fuzz
            ));
            *attenuation = hit_result->albedo;
            break;

        case DIELECTRIC:
            refraction_index = hit_result->front_face ? 
                ( 1.0F/(hit_result->refraction_index) ) :
                hit_result->refraction_index;
            cos_theta = fmin(
                dot(negate(vec3_unit(ray_in->direction)), hit_result->normal), 
                1.0F
            );
            sin_theta = sqrt(1.0 - cos_theta*cos_theta);
        
            // Cannot refract, so it reflects (total internal reflection)
            // Reflectivity varying based on the angle is given by Shlick's 
            // Approximation
            cannot_refract = ((refraction_index*sin_theta) > 1.0F);
            scatter_direction = (cannot_refract || \
            reflectance(cos_theta, refraction_index) > h_random_float()) ?
                REFLECT(vec3_unit(ray_in->direction), hit_result->normal) :
                refract(vec3_unit(ray_in->direction), 
                            vec3_unit(hit_result->normal), 
                            refraction_index);

            *attenuation = color_new(1.0F, 1.0F, 1.0F);
            break;

        default:
            fprintf(stderr, "Material not yet implemented %d! Aborting", 
                hit_result->surface_material);
            exit(1);
        }

        *ray_scattered = ray_new(hit_result->p, scatter_direction);
}
__device__ void d_scatter(
    t_hit_result *hit_result,
    t_ray *ray_in, 
    t_color *attenuation, 
    t_ray *ray_scattered,
    curandState *state
) {

    t_vec3 scatter_direction;
    float refraction_index, cos_theta, sin_theta;
    bool cannot_refract;
    switch (hit_result->surface_material) {

        case LAMBERTIAN:
            scatter_direction = sum(
                hit_result->normal, 
                d_vec3_random_unit(state)
            );
            scatter_direction = NEAR_ZERO(scatter_direction) ?
                                    hit_result->normal : scatter_direction;
            *attenuation = hit_result->albedo;
            break;

        case METAL:
            scatter_direction = REFLECT(ray_in->direction, hit_result->normal);
            scatter_direction = sum(
                scatter_direction, 
                scale(d_vec3_random_unit(state), hit_result->fuzz)
            );
            *attenuation = hit_result->albedo;
            break;

        case DIELECTRIC:
            refraction_index = hit_result->front_face ? 
                ( 1.0F/(hit_result->refraction_index) ) :
                hit_result->refraction_index;
            cos_theta = fmin(
                dot(negate(vec3_unit(ray_in->direction)), hit_result->normal), 
                1.0F
            );
            sin_theta = sqrt(1.0F - cos_theta*cos_theta);
        
            // Cannot refract, so it reflects (total internal reflection)
            // Reflectivity varying based on the angle is given by Shlick's 
            // Approximation
            cannot_refract = ((refraction_index*sin_theta) > 1.0F);
            scatter_direction = (cannot_refract || \
                                 reflectance(cos_theta, refraction_index) \
                                 > d_random_float(state)) ?
                REFLECT(vec3_unit(ray_in->direction), hit_result->normal) :
                refract(vec3_unit(ray_in->direction), 
                            vec3_unit(hit_result->normal), 
                            refraction_index);

            *attenuation = color_new(1.0F, 1.0F, 1.0F);
            break;

        default:
            // TODO Find a way to log to stderr from device code
            return;
        }

        *ray_scattered = ray_new(hit_result->p, scatter_direction);
}

#endif
