#include <stdio.h>
#include <stdlib.h>

#include "color.h"
#include "common.h"
#include "hittable.h"
#include "point3.h"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"

#define RAY_T_MAX 9999
#define NUMBER_OF_SPHERES 2

sphere world[NUMBER_OF_SPHERES] = {
    //   center,   radius
    { {0,0,-1}, 0.5 },
    { {0,-100.5,-1}, 100 },
};

// Determining a different color for each of the rays that hit the screen, in 
// case based on blending blue and white
color ray_color(ray r) {
    
    hit_result temp, result;
    bool hit_anything = false;
    my_decimal closest_hit = RAY_T_MAX;
    // For each sphere
    for (int i = 0; i < NUMBER_OF_SPHERES; i++) {
        // If the ray hits the sphere before all the other spheres
        temp = sphere_hit(r, world[i], 0, closest_hit);
        if (temp.did_hit) {
            hit_anything = true;
            closest_hit = temp.t;
            result = temp;
        } 
    }

    if (result.did_hit) {
        // Normal vector returned as a color
        vec3 n = result.normal;
        return vec3_scale(
            color_new(n.x + 1, n.y + 1, n.z + 1),
            0.5);
    }

    vec3 unit_direction = vec3_unit(r.direction);
    my_decimal a = 0.5*(unit_direction.y + 1.0);
    return vec3_sum(
                vec3_scale(color_new(1.0, 1.0, 1.0),  (1.0 - a)),
                vec3_scale(color_new(0.5, 0.7, 1.0), a)
    );
}

int main() {

    // Set image width and height in pixels
    my_decimal aspect_ratio = 16.0/9.0;
    int image_width = 400;
    int image_height = (int) (image_width/aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // Initialize focal length, viewport size and camera center

    // Focal length is the distance between the center of the viewport and the 
    // camera center
    my_decimal focal_length = 1.0;
    my_decimal viewport_height = 2.0;
    my_decimal viewport_width = viewport_height * (((double) image_width)/image_height);

    // The point of the space from which all rays originate. Conventionally, 
    // its coordinates are fixed to (0, 0, 0).
    // The camera center is positioned on the line orthogonal to the viewport
    // center, at the focal distance.
    point3 camera_center = point3_new(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical 
    // viewport edges.
    vec3 viewport_u = vec3_new(viewport_width, 0, 0);
    vec3 viewport_v = vec3_new(0, -viewport_height, 0);
    vec3 pixel_delta_u = vec3_divide(viewport_u, image_width);
    vec3 pixel_delta_v = vec3_divide(viewport_v, image_height);

    // Location of upper left pixel
    point3 pixel00 = camera_center;
    pixel00 = vec3_subtract(pixel00, vec3_new(0, 0, focal_length));
    pixel00 = vec3_subtract(pixel00, vec3_divide(viewport_u, 2));
    pixel00 = vec3_subtract(pixel00, vec3_divide(viewport_v, 2));
    pixel00 = vec3_sum(pixel00, vec3_scale(
        vec3_sum(pixel_delta_u, pixel_delta_v), 0.5));

    // Write PPM header to stdout
    printf("P3\n%d %d\n255\n", image_width, image_height);

    // Render the image
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {

            // Current pixel we are considering
            point3 pixel_center = vec3_sum(pixel00, vec3_scale( pixel_delta_u, i));
            pixel_center = vec3_sum(pixel_center, vec3_scale( pixel_delta_v, j));

            // Vector connecting camera center to the current pixel, used to 
            // instantiate the ray. The direction is not normalized to increase
            // performance.
            vec3 ray_direction = vec3_subtract(pixel_center, camera_center);
            ray r = ray_new(camera_center, ray_direction);

            color c = ray_color(r);
            color_print_PPM(c);
        }
        // Update progress counter
        fprintf(stderr, "\rScanlines processed: %d/%d", j + 1, image_height);
        fflush(stderr);
    }

    fprintf(stderr, "\rDone.                                    \n");
    return 0;
}
