#ifndef COLOR_H
#define COLOR_H

#include <stdio.h>

#include "common.h"
#include "vec3.h"

// A color is represented as a vector with three values normalized in the range
// [0, 1]
#define t_color t_vec3
#define color_new vec3_new

#define COLOR_BLACK  (t_color)   {0.0F, 0.0F, 0.0F}
#define COLOR_GRAY   (t_color)   {0.5F, 0.5F, 0.5F}
#define COLOR_WHITE  (t_color)   {1.0F, 1.0F, 1.0F}
#define COLOR_RED    (t_color)   {1.0F, 0.0F, 0.0F}
#define COLOR_BLUE   (t_color)   {0.0F, 0.0F, 1.0F}
#define COLOR_GREEN  (t_color)   {0.0F, 1.0F, 0.0F}
#define COLOR_SKY    (t_color)   {0.5F, 0.7F, 1.0F}

// Return a blend (lerp) going  from  color1 and color2 based on blend factor a
#define BLEND(a, color1, color2) \
    sum(scale(color1, (0.5F * (1.0F - a))), \
    scale(color2, (0.5F * (1.0F + a))))

#define TO_GAMMA(a) (a > 0.0F) ? sqrt(a) : a

// Prints color to stderr in readable format
__host__ void color_print(t_color c) {
    fprintf(stderr, "%d %d %d\n", 
        (int) (255.999F * c.x), 
        (int) (255.999F * c.y), 
        (int) (255.999F * c.z)
    );
}

__host__ __device__ void color_write_at(
    t_color c, 
    long offset, 
    unsigned char *addr
) {
    // Clamp color RGB components to interval [0, 0.999]
    c.x = (c.x > 0.999F) ? 0.999F : c.x;
    c.y = (c.y > 0.999F) ? 0.999F : c.y;
    c.z = (c.z > 0.999F) ? 0.999F : c.z;
    c.x = (c.x < 0.0F) ? 0.0F : c.x;
    c.y = (c.y < 0.0F) ? 0.0F : c.y;
    c.z = (c.z < 0.0F) ? 0.0F : c.z;
     
    float r = TO_GAMMA(c.x);
    float g = TO_GAMMA(c.y);
    float b = TO_GAMMA(c.z);

    addr[offset] = (unsigned char) (255.999F * r);
    addr[offset+1] = (unsigned char) (255.999F * g);
    addr[offset+2] = (unsigned char) (255.999F * b);
}

#endif 