#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"
#include "common.h"
#include "ray.h"

typedef enum {
    LAMBERTIAN, 
    METAL,
    DIELECTRIC
} material_type;

typedef struct {
    material_type type;

    t_color albedo; // For lambertian and metal
    float fuzz; // For metal
    float refraction_index; // For dielectric
} t_material;

__host__ __device__ t_material new_lambertian(t_color albedo) {
    t_material material;
    material.type = LAMBERTIAN;
    material.albedo = albedo;
    material.fuzz = -1.0F;
    material.refraction_index = -1.0F;
    return material;
}

__host__ __device__ t_material new_metal(t_color albedo, float fuzz) {
    t_material material;
    material.type = METAL;
    material.albedo = albedo;
    material.fuzz = (fuzz < 1.0F) ? fuzz : 1.0F;
    material.refraction_index = -1.0F;
    return material;
}

__host__ __device__ t_material new_dielectric(float refraction_index) {
    t_material material;
    material.type = DIELECTRIC;
    material.albedo = (t_color){-1.0F, -1.0F, -1.0F};
    material.fuzz = -1;
    material.refraction_index = refraction_index;
    return material;
}

// Reflectance of a dielectric
__host__ __device__ float reflectance(
    float cosine, 
    float refraction_index
) {
    float r0 = (1.0F - refraction_index) / (1.0F + refraction_index);
    r0 = r0*r0;
    return r0 + (1.0F-r0)*pow((1.0F - cosine), 5);
}


#endif 