#ifndef PARAMETERS_H
#define PARAMETERS_H

#define my_decimal double

// Minimum value that is safe to square without underflow
#define MY_DECIMAL_UNDERFLOW_LIMIT 1e-18

#define SAMPLES_PER_PIXEL 100
#define MAX_RAY_BOUNCES 100

// Lambertian reflection models the distribution using wich way rays are 
// reflected from a diffuse surface
#define USE_LAMBERTIAN_REFLECTION true

#endif