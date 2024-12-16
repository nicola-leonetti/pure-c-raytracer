#ifndef POINT3_H
#define POINT3_H

#include "common.h"
#include "vec3.h"

#define point3 vec3

#define point3_new vec3_new

void point3_print(const point3 v) {
    printf("Point(%f, %f, %f)\n", v.x, v.y, v.z);
}

#endif