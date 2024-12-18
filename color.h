#ifndef COLOR_H
#define COLOR_H

#include <stdio.h>

#include "common.h"
#include "vec3.h"

// A color is represented as a vector with three values normalized in the range
// [0, 1]
#define color vec3
#define color_new vec3_new

#define COLOR_BLACK color_new(0.0, 0.0, 0.0)
#define COLOR_WHITE color_new(1.0, 1.0, 1.0)
#define COLOR_BLUE color_new(0.5, 0.7, 1.0)

// Return a blend (lerp) going  from  color1 and color2 based on blend factor a
#define BLEND(a, color1, color2) \
        sum(scale(color1, (0.5 * (1.0 - a))), \
        scale(color2, (0.5 * (1.0 + a))))

// Prints color to stderr in readable format
void color_print(color c) {
    fprintf(stderr, "%d %d %d\n", 
            (int) (255.999 * c.x), 
            (int) (255.999 * c.y), 
            (int) (255.999 * c.z));
}

// Prints the color to stdout in PPM format
void color_print_PPM(color c) {
    // Clamp color RGB components to interval [0, 0.999]
    c.x = (c.x > 0.999) ? 0.999 : c.x;
    c.y = (c.y > 0.999) ? 0.999 : c.y;
    c.z = (c.z > 0.999) ? 0.999 : c.z;
    c.x = (c.x < 0) ? 0 : c.x;
    c.y = (c.y < 0) ? 0 : c.y;
    c.z = (c.z < 0) ? 0 : c.z;

    printf("%d %d %d\n", 
            (int) (255.999 * c.x), 
            (int) (255.999 * c.y), 
            (int) (255.999 * c.z));
}

#endif 