#include "camera.h"
#include "sphere.h"

#define ASPECT_RATIO 16.0/9.0
#define VIEWPORT_WIDTH 400

#define NUMBER_OF_SPHERES 2
sphere world[NUMBER_OF_SPHERES] = {
    //   center,   radius
    { {0,0,-1}, 0.5 },
    { {0,-100.5,-1}, 100 },
};

int main() {
    // Initialize RNG
    srand((unsigned int) time(NULL));
    
    camera cam = camera_new(ASPECT_RATIO, VIEWPORT_WIDTH);
    camera_render(cam, world, NUMBER_OF_SPHERES);
}
