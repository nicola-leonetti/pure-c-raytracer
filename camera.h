#ifndef CAMERA_H
#define CAMERA_H

#include "color.h"
#include "common.h"
#include "point3.h"
#include "sphere.h"
#include "vec3.h"

#define RAY_T_MAX 9999

typedef struct {
    my_decimal aspect_ratio;
    int image_width, image_height;

    // Camera center is the point of the space from which all rays originate. 
    // Conventionally set to (0, 0, 0).
    // It lies on the line orthogonal to the viewport center, at the focal 
    // distance.
    point3 center;

    // Distance between the center of the viewport and the camera center
    my_decimal focal_length;

    // The viewport is a virtual rectangle in our 3D world containing the 
    // square pixels in our rendered image.
    // One or more rays are sent from the camera center through each pixel of 
    // the viewport and the intersection point with the objects in the scene is 
    // determined, which in turn determines the pixel's color.
    my_decimal viewport_height, viewport_width;

    // Convenient vectors to navigate the viewport through its width and down 
    // its height.
    vec3 viewport_u, viewport_v;
    vec3 pixel_delta_u, pixel_delta_v;

    // Center of upper-left pixel in the viewpoer
    point3 pixel00;
} camera;

// Constructor
camera camera_new(my_decimal aspect_ratio, int image_width) {
    camera cam;

    cam.image_height = (int) (image_width/aspect_ratio);
    cam.image_height = (cam.image_height < 1) ? 1 : cam.image_height;
    cam.image_width = image_width;

    cam.center = point3_new(0, 0, 0);
    cam.focal_length = 1.0;

    cam.viewport_height = 2.0;
    cam.viewport_width = cam.viewport_height * 
        (((my_decimal) image_width)/cam.image_height);
    cam.viewport_u = vec3_new(cam.viewport_width, 0, 0);
    cam.viewport_v = vec3_new(0, -cam.viewport_height, 0);
    cam.pixel_delta_u = divide(cam.viewport_u, cam.image_width);
    cam.pixel_delta_v = divide(cam.viewport_v, cam.image_height);

    cam.pixel00 = cam.center;
    cam.pixel00 = subtract(cam.pixel00, vec3_new(0, 0, cam.focal_length));
    cam.pixel00 = subtract(cam.pixel00, divide(cam.viewport_u, 2));
    cam.pixel00 = subtract(cam.pixel00, divide(cam.viewport_v, 2));
    cam.pixel00 = sum(cam.pixel00, scale(
        sum(cam.pixel_delta_u, cam.pixel_delta_v), 0.5));

    return cam;
}

// Return a random ray with the end in a random point inside the (i, j) pixel.
ray get_random_ray(camera *cam, int i, int j) {
    // Offset from the center of the vector generated in the unit square
    // [-0.5, 0.5]x[-0.5, 0.5]
    vec3 offset  = vec3_new(random_my_decimal() - 0.5, 
                            random_my_decimal() - 0.5, 0.0);

    // Use the offset to select a random point inside the (i, j) pixel
    point3 pixel_sample = cam->pixel00;
    pixel_sample = sum(pixel_sample, scale(cam->pixel_delta_u, i+offset.x));
    pixel_sample = sum(pixel_sample, scale(cam->pixel_delta_v, j+offset.y));

    vec3 ray_direction = subtract(pixel_sample, cam->center); 
    return ray_new(cam->center, ray_direction);   
}

// Determining a different color for each of the pixels of the viewport by 
// sending one or more rays from the camera center to each pixel 
color ray_color(ray *r, sphere world[], int number_of_spheres, int bounces) {
    hit_result temp, result;
    bool hit_anything = false;
    my_decimal closest_hit = RAY_T_MAX;

    // For each sphere, if the ray hits the sphere before all the other spheres
    // (0.001 lower bound is used to fix "shadow acne")
    for (int i = 0; i < number_of_spheres; i++) {
        temp = sphere_hit(r, world[i], 0.001, closest_hit);
        if (temp.did_hit) {
            hit_anything = true;
            closest_hit = temp.t;
            result = temp;
        } 
    }

    if (result.did_hit) {
        // Return the color of a random ray. To limit recursion depth, after a 
        // certain number of recursion, black is returned.
        #if USE_LAMBERTIAN_REFLECTION 
            vec3 direction = sum(result.normal, vec3_random_unit());
        #else
            vec3 direction = vec3_random_on_hemisphere(result.normal);
        #endif

        ray reflected_ray = ray_new(result.p, direction);
        return (bounces == 0) ? COLOR_BLACK :
                                scale(ray_color(&reflected_ray,
                                    world, number_of_spheres, bounces-1), 0.5);
    }

    // If no object is hit, return a blend between blue and white based on the 
    // y coordinate, so going vertically from white all the way to blue
    return BLEND(vec3_unit(r->direction).y, COLOR_WHITE, COLOR_BLUE);
}

void camera_render(camera cam, sphere world[], int number_of_spheres) {
    // PPM header
    printf("P3\n%d %d\n255\n", cam.image_width, cam.image_height);

    // Render cycle
    for (int j = 0; j < cam.image_height; j++) {
        for (int i = 0; i < cam.image_width; i++) {
            color pixel_color = color_new(0, 0, 0);

            // Antialiasing: sample SAMPLE_PER_PIXEL colors and average them to 
            // obtain pixel color
            for (int sample = 0; sample < SAMPLES_PER_PIXEL; sample++) {
                ray random_ray = get_random_ray(&cam, i, j);
                color sampled_color = ray_color(&random_ray, world, 
                                        number_of_spheres, MAX_RAY_BOUNCES);
                pixel_color = sum(pixel_color, sampled_color);
            } 
            pixel_color = divide(pixel_color, SAMPLES_PER_PIXEL);

            color_print_PPM(pixel_color);
        }
        // Update progress counter
        fprintf(stderr, "\rScanlines processed: %d/%d", j + 1, cam.image_height);
        fflush(stderr);
    }

    fprintf(stderr, "\rDone.                                    \n");
}

#endif