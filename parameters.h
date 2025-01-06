#ifndef PARAMETERS_H
#define PARAMETERS_H

#define RNG_SEED 1234

// CUDA block size
dim3 block(8,8); 

// Minimum value that is safe to square without underflow
#define float_UNDERFLOW_LIMIT 1e-18F

#define NEAR_ZERO_TRESHOLD 1e-8F

#define ASPECT_RATIO 16.0F/9.0F
#define VIEWPORT_WIDTH 600 // 1920 for HD

// Points from thich the camera looks from and towards which it looks at
#define LOOK_FROM {20.0F, 4.0F, 20.0F}
#define LOOK_AT {0.0F, 0.0F, 0.0F}

// What the camera considers to be "up"
#define UP_DIRECTION {0, 1, 0}

// Variation angle of rays through each pixel
#define DEFOCUS_ANGLE 0.0F

// How distant a plane with perfect focus is from the lookfrom point
#define FOCUS_DISTANCE 20.0F

// Vertical field of view, aka the angle between the image top and bottom.
// Expressed in degrees
#define VERTICAL_FOV_DEGREES 20.0F

// How many rays to send inside each square pixel in order to obtain a more 
// precise color (antialiasing).
#define SAMPLES_PER_PIXEL 100

// Limit to how many times a ray can bounce from a surface to another before 
// determining its color. This is done in order to avoid stack overflow
#define MAX_RAY_BOUNCES 10

#endif