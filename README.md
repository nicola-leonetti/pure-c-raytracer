# SDM-raytracer

Simple raytracer written in CUDA C, based on the the book [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

## Compiling and running locally
The following compiles and runs the program with some optimizations, creating the image file `img.ppm`.

```bash
nvcc -G --use_fast_math -rdc=true -arch=sm_86 --maxrregcount 40  main.cu -o main
./main > img.ppm
```

The compilation flags are there to optimize performance by reducing the number of used registers on the GPU. 
Please note that the code was tested and optimized on a NVIDIA GeForce RTX 4060 Ti, so optimal flags for other GPUs may be different.

## Compiling with different parameters
The behavior of the raytracer can be changed by editing the header file 
`parameters.h`. Here are the most important settings:

### General settings
- `USE_CUDA`: Defines whether the code runs on CPU or Nvidia GPU (CUDA).
- `NUMBER_OF_SPHERES`: Total number of spheres rendered in the scene.
### Camera settings
- `ASPECT_RATIO`: The aspect ratio of the rendered image.
- `VIEWPORT_WIDTH`: Width of the viewport in pixels.
- `LOOK_FROM`: The position of the camera in 3D space.
- `LOOK_AT`: The point in 3D space where the camera is aimed.
- `UP_DIRECTION`: The direction considered "up" by the camera.
- `DEFOCUS_ANGLE`: Variation angle of rays through each pixel.
- `FOCUS_DISTANCE`: Distance from the `LOOK_FROM` point to the plane of perfect focus.
- `VERTICAL_FOV_DEGREES`: Vertical field of view in degrees.
### Raytracing settings
- `SAMPLES_PER_PIXEL`: Number of rays sent per pixel for antialiasing.
- `MAX_RAY_BOUNCES`: Maximum number of ray bounces before terminating.

By modifying these parameters, you can customize the raytracer to better suit your needs.
