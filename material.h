#ifndef MATERIAL_H
#define MATERIAL_H

#include "common.h"

typedef struct {
    // How much light is reflected
    my_decimal albedo;
} t_material;

#define LAMBERTIAN ( (t_material) {0.5} )

#endif 