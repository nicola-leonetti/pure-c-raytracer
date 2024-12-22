#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"
#include "common.h"
#include "ray.h"

typedef enum {
    LAMBERTIAN, 
    METAL,
    DIELECTRIC
} t_material;

// Reflectance of a dielectric
my_decimal reflectance(my_decimal cosine, my_decimal refraction_index) {
    my_decimal r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine), 5);
}


#endif 