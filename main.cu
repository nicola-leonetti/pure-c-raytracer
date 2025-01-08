#include <time.h>

#include "camera.h"
#include "color.h"
#include "common.h"
#include "material.h"
#include "sphere.h"

__host__ double h_cpu_second() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double) ts.tv_sec + (double) ts.tv_nsec * 1.e-9F);
}

__host__ void h_init_world(t_sphere *world) {
    fprintf(stderr, "Initializing spheres...");

    // Ground sphere (Lambertian material)
    world[0] = sphere_new(
        point3_new(0.0F, -1000.0F, 0.0F), 1000.0F, new_lambertian(COLOR_GRAY));

    world[1] = sphere_new(point3_new(0.0F, 1.0F, 0.0F), 1.0F, new_dielectric(1.5F));
    world[2] = \
        sphere_new(point3_new(-4.0F, 1.0F, 0.0F), 1.0F, new_lambertian(COLOR_BLUE));
    world[3] = \
        sphere_new(point3_new(4.0F, 1.0F, 0.0F), 1.0F, new_metal(COLOR_GREEN, 0.0F));

    // Create a grid of random spheres
    int index = 4;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            // Randomize material choice
            float choose_mat = h_random_float();
            t_point3 center = point3_new(
                a + 0.9F * h_random_float(), 
                0.2F, 
                b + 0.9F * h_random_float()
            );
            t_material sphere_material;

            if (choose_mat < 0.8F) {
                // Lambertian (diffuse)
                sphere_material = new_lambertian(
                    color_new(h_random_float()*h_random_float(), 
                              h_random_float()*h_random_float(), 
                              h_random_float()*h_random_float()
                            ));
                world[index++] = sphere_new(center, 0.2F, sphere_material);
            } 
            else if (choose_mat < 0.95F) {
                // Metal
                t_color color = color_new(
                    h_random_float_in(0.5F, 1.0F), 
                    h_random_float_in(0.5F, 1.0F), 
                    h_random_float_in(0.5F, 1.0F)
                );
                sphere_material = new_metal(
                    color, 
                    h_random_float_in(0.0F, 0.5F)
                );
                world[index++] = sphere_new(center, 0.2F, sphere_material);
            } 
            else {
                // Dielectric (glass)
                world[index++] = sphere_new(center, 0.2F, new_dielectric(1.5F));
            }
            
        }
    }

    fprintf(stderr, "\r                            \r");
    fprintf(stderr, "Spheres initilized\n");
}

__host__ void h_write_PPM_img_to_stdout(
    unsigned char *img, 
    int width, 
    int height
) {
    // PPM header
    printf("P3\n%d %d\n255\n", width, height);
    for (int pixel = 0; pixel < width*height*3; pixel+=3) {
        printf("%d %d %d\n", img[pixel], img[pixel+1], img[pixel+2]);
    }
}


int main() {
    #if USE_CUDA
    print_device_info(0);
    #endif

    // Initialize RNG
    srand((unsigned int) RNG_SEED);

    double start, end;
    
    start = h_cpu_second();
    // Initialize spheres on host
    int world_size = NUMBER_OF_SPHERES*sizeof(t_sphere);
    t_sphere *h_world = (t_sphere*) malloc(world_size); 
    h_init_world(h_world);

    #if USE_CUDA
    // Copy spheres host -> device
    t_sphere *d_world;
    CHECK(cudaMalloc((void**)&d_world, world_size));
    CHECK(cudaMemcpy(d_world, h_world, world_size, cudaMemcpyHostToDevice));
    #endif

    // TODO Piccola ottimizzazione creare camera direttamente su device
    t_camera cam = camera_new(ASPECT_RATIO, 
                              VIEWPORT_WIDTH, 
                              VERTICAL_FOV_DEGREES,
                              (t_point3) LOOK_FROM, 
                              (t_point3) LOOK_AT,
                              DEFOCUS_ANGLE,
                              FOCUS_DISTANCE);
    #if USE_CUDA
    t_camera *h_cam = &cam;
    t_camera *d_cam;
    CHECK(cudaMalloc((void**)&d_cam, sizeof(cam)));
    CHECK(cudaMemcpy(d_cam, h_cam, sizeof(cam), cudaMemcpyHostToDevice));
    #endif

    // Allocate on device one RNG state for each pixel
    #if USE_CUDA
    int number_of_pixels = cam.image_width*cam.image_height;
    curandState *d_random_states;
    CHECK(cudaMalloc(
        (void**) &d_random_states, 
        number_of_pixels*sizeof(curandState)
    ));
    #endif

    // Allocate space for the image on host and device
    long img_size = cam.image_height*cam.image_width*sizeof(unsigned char)*3;
    unsigned char *h_result_img = (unsigned char*) malloc(img_size);
    
    #if USE_CUDA
    unsigned char *d_result_img;
    CHECK(cudaMalloc((void**)&d_result_img, img_size));
    CHECK(cudaMemcpy(
        d_result_img, 
        h_result_img, 
        img_size, 
        cudaMemcpyHostToDevice
    ));
    #endif
    end = h_cpu_second();

    double init_time = end-start;

    fprintf(
        stderr, 
        "Image size: %dx%d, %d channels, %ld bytes\n", 
        cam.image_height, 
        cam.image_width,
        3,
        img_size
    );

    #if USE_CUDA
    fprintf(
        stderr,
        "Launching render kernel with 2D grid shape (%u, %u)\n", 
        (cam.image_width + block.x - 1) / block.x, 
        (cam.image_height + block.y - 1) / block.y
    );
    #endif

    start = h_cpu_second();

    // CUDA version
    #if USE_CUDA
    dim3 grid(
        (cam.image_width + block.x - 1) / block.x, 
        (cam.image_height + block.y - 1) / block.y
    );
    d_camera_render<<<grid, block>>>(
        d_cam,
        d_world,
        NUMBER_OF_SPHERES,
        d_result_img,
        d_random_states
    );
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
    #else
    // CPU version
    start = h_cpu_second();
    h_camera_render(&cam, h_world, NUMBER_OF_SPHERES, h_result_img);
    #endif

    end = h_cpu_second();

    double render_time = end-start;

    start = h_cpu_second();
    #if USE_CUDA
    CHECK(cudaMemcpy(
        h_result_img, 
        d_result_img, 
        img_size, 
        cudaMemcpyDeviceToHost
    ));
    #endif
    h_write_PPM_img_to_stdout(h_result_img, cam.image_width, cam.image_height);
    end = h_cpu_second();

    double copy_back_time = end-start;

    fprintf(stderr, "Initialization time: %.6fs\n", init_time);
    fprintf(stderr, "Render time: %.6fs\n", render_time);
    fprintf(stderr, "Copy back time: %.6fs\n", copy_back_time);
    
    #if USE_CUDA
    CHECK(cudaFree(d_world));
    CHECK(cudaFree(d_cam));
    CHECK(cudaFree(d_result_img));
    #endif

    free(h_world);
    free(h_result_img);
}
