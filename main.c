#include <time.h>

#include "camera.h"
#include "color.h"
#include "common.h"
#include "material.h"
#include "sphere.h"

#define NUMBER_OF_SPHERES 4
t_sphere world[NUMBER_OF_SPHERES] = {
    // {
    //     {0,0,-1},    // center
    //     0.5,         // radius
    //     COLOR_RED,  // color
    //     METAL   // type of material
    // },
    // {
    //     {0,-100.5,-1}, 
    //     100, 
    //     COLOR_GRAY,
    //     LAMBERTIAN 
    // },
    {
        {0,-100.5,-1.0}, 
        100, 
        {0.8,0.8,0},
        LAMBERTIAN,
        1.0
    },
    {
        {0,0,-1.2}, 
        0.5, 
        {0.1,0.2,0.5},
        LAMBERTIAN,
        1.0
    },
    {
        {-1,0,-1}, 
        0.5, 
        {0.8,0.8,0.8},
        METAL,
        0.3
    },
    {
        {1,0,-1}, 
        0.5, 
        {0.8,0.6,0.2},
        METAL,
        1.0
    },
};

int main() {
    // Initialize RNG
    srand((unsigned int) time(NULL));

    clock_t start = clock();
    
    t_camera cam = camera_new(ASPECT_RATIO, VIEWPORT_WIDTH);
    camera_render(&cam, world, NUMBER_OF_SPHERES);

    clock_t end  = clock();
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    fprintf(stderr, "Elapsed time: %.6fs\n", elapsed_time);
}
