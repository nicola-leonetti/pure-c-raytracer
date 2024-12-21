#ifndef PARAMETERS_H
#define PARAMETERS_H

#define my_decimal double

// Minimum value that is safe to square without underflow
#define MY_DECIMAL_UNDERFLOW_LIMIT 1e-18

#define ASPECT_RATIO 16.0/9.0
#define VIEWPORT_WIDTH 400

// How many rays to send inside each square pixel in order to obtain a more 
// precise color (antialiasing).
#define SAMPLES_PER_PIXEL 10

// Limit to how many times a ray can bounce from a reflective surface to 
// another before determining its color. This is done in order to avoid stack 
// overflow
#define MAX_RAY_BOUNCES 100

// Lambertian reflection models the distribution using wich way rays are 
// reflected from a diffuse surface
#define USE_LAMBERTIAN_REFLECTION true

#endif