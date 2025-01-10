# SDM-raytracer

Simple raytracer written in pure C, based on the book [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

## Compiling and running locally
The following compiles and runs the program, creating the image file `img.ppm` in the current directory.

```bash
nvcc -G --use_fast_math -rdc=true -arch=sm_86 --maxrregcount 40  main.cu -o coalesc-400
./main > img.ppm
```
