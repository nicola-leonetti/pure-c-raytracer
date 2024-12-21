#include <time.h>

#include "camera.h"
#include "common.h"
#include "sphere.h"

#define NUMBER_OF_SPHERES 2
t_sphere world[NUMBER_OF_SPHERES] = {
    //   center,   radius
    { {0,0,-1}, 0.5 },
    { {0,-100.5,-1}, 100 },
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
