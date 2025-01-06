#ifndef COMMON_H
#define COMMON_H

#include "parameters.h"

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdbool.h>
#include <stdlib.h>

#define PI 3.14159265358979323846F

// Returns a float in [0, 1)
__host__ float h_random_float() {
    return ((float) rand()) / ((float) RAND_MAX + 1.0F);
}
__device__ float d_random_float(curandState *state) {
    return (float) curand_uniform(state);
}

// Returns a float in [min, max)
__host__ float h_random_float_in(float min, float max) {
    return min + (max-min)*h_random_float();
}
__device__ float d_random_float_in(
    float min, 
    float max, 
    curandState *state
) {
    return min + (max-min)*d_random_float(state);
}

__host__ __device__ float degrees_to_radians(float deg) {
    return (deg*PI) / 180.0F;
}

// CUDA macro for checking function calls
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf( \
            stderr, \
            "code: %d, reason: %s\n", \
            error, \
            cudaGetErrorString(error) \
        ); \
        exit(1); \
    } \
}

__host__ inline void print_device_info(int id) {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, id);
    
    fprintf(stderr, "\n-----------------Device properties-----------------\n");
    fprintf(stderr, "GPU name: %s \n", properties.name);
    fprintf(
        stderr, 
        "Compute capability: %d.%d\n", 
        properties.major,
        properties.minor
    );
    fprintf(
        stderr, 
        "Max number of threads per block: %d \n", 
        properties.maxThreadsPerBlock
    );
    fprintf(
        stderr, 
        "Max size of a block of threads: (%d, %d, %d) \n", 
        properties.maxThreadsDim[0],
        properties.maxThreadsDim[1],
        properties.maxThreadsDim[2]
    );
    fprintf(
        stderr, 
        "Max size of grid of blocks: (%d, %d, %d) \n",
        properties.maxGridSize[0],
        properties.maxGridSize[1],
        properties.maxGridSize[2]
    );
    fprintf(stderr, "---------------------------------------------------\n\n");
}

#endif