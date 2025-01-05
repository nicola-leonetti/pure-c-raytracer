# SDM-raytracer

Simple raytracer written in pure C, based on the book [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

## Compiling and running locally
The following compiles and runs the program, creating the image file `img.ppm` in the current directory.

```bash
gcc main.c -o main -lm
./main > img.ppm
```

## A note about host and device functions naming conventions
In this project, the following conventions are used:
- host-only functions have a name starting with the prefix `h_` and are always annotated with `__host__` even if CUDA doesn't mandate it.
- device-only functions follow use `d_` and `__device__` instead.
- host and device functions have no prefix in their name and use the double annotation `__host__ __device__`. To enforce reusability, a function is considered a host and device function regardless of it being actually used in both ways in the code for the mere fact of sharing the same C/CUDA code in its implementation.
