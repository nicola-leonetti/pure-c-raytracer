#ifndef COMMON_H
#define COMMON_H

#include "parameters.h"

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdbool.h>
#include <stdlib.h>

#define PI 3.14159265358979323846

// Returns a my_decimal in [0, 1)
__host__ my_decimal h_random_my_decimal() {
    return ((my_decimal) rand()) / ((my_decimal) RAND_MAX + 1);
}
__device__ my_decimal d_random_my_decimal(curandState *state) {
    return (my_decimal) curand_uniform(state);
}

// Returns a my_decimal in [min, max)
__host__ my_decimal h_random_my_decimal_in(my_decimal min, my_decimal max) {;
    return min + (max-min)*h_random_my_decimal();
}
__device__ my_decimal d_random_my_decimal_in(my_decimal min, my_decimal max, curandState *state) {;
    return min + (max-min)*d_random_my_decimal(state);
}

__host__ __device__ my_decimal degrees_to_radians(my_decimal deg) {
    return (deg*PI) / 180.0;
}

// CUDA macro for checking function calls
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

__host__ inline void print_device_info(int id) {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    
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