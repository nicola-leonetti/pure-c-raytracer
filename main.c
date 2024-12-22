#include <time.h>

#include "camera.h"
#include "color.h"
#include "common.h"
#include "material.h"
#include "sphere.h"

// TODO Take this from a file
#define NUMBER_OF_SPHERES 5
t_sphere world[NUMBER_OF_SPHERES] = {
    { // Ground big sphere
        {0,-100.5,-1.0}, // center
        100, // radius
        {0.8,0.8,0.0}, // color 
        LAMBERTIAN, // material
        1.0, // fuzz
        1.50 // refraction index
    },
    { // Center blue sphere
        {0,0,-1.2}, 
        0.5, 
        {0.1,0.2,0.5},
        LAMBERTIAN,
        1.0,
        -1
    },
    { // Left sphere
        {-1,0,-1}, 
        0.5, 
        {0.8,0.8,0.8},
        DIELECTRIC,
        -1,
        1.50
    },
    { // Bubble inside left sphere
        {-1,0,-1}, 
        0.4, 
        {0.8,0.8,0.8},
        DIELECTRIC,
        -1,
        1.0/1.50
    },
    { // Right sphere
        {1,0,-1}, 
        0.5, 
        {0.8,0.6,0.2},
        METAL,
        1.0,
        -1
    },
};

int main() {
    // Initialize RNG
    srand((unsigned int) time(NULL));

    clock_t start = clock();

    t_camera cam = camera_new(ASPECT_RATIO, 
                              VIEWPORT_WIDTH, 
                              VERTICAL_FOV_DEGREES,
                              point3_new(-2, 2, 1), point3_new(0, 0, -1),
                              10.0,
                              3.4);
    camera_render(&cam, world, NUMBER_OF_SPHERES);

    clock_t end  = clock();
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    fprintf(stderr, "Elapsed time: %.6fs\n", elapsed_time);
}
