#ifndef CAMERA_H
#define CAMERA_H

#include "color.h"
#include "common.h"
#include "point3.h"
#include "ray.h"
#include "parameters.h"
#include "sphere.h"
#include "vec3.h"

#define RAY_T_MAX 9999

typedef struct {
    int image_width, image_height;

    // Point of space from which rays originate
    t_point3 center;
    
    // The viewport is a virtual rectangle in our 3D world containing the 
    // square pixels that make up the rendered image.
    // One or more rays are sent from the camera center through each pixel of 
    // the viewport and the intersection point with the objects in the scene is 
    // determined, which in turn determines the pixel's color.
    // Vectors used to navigate the viewpoer through its width and down its 
    // height
    t_vec3 pixel_delta_u, pixel_delta_v;

    // Center of upper-left pixel in the viewport
    t_point3 pixel00;

    // Used later to determine ray origin
    bool defocus_angle_is_negative;

    // Horizontal and vertical radius of defocus disk
    t_vec3 defocus_disk_u, defocus_disk_v;
} t_camera;

// Constructor
t_camera camera_new(
    float aspect_ratio, 
    int image_width, 
    float vertical_fov,
    t_point3 look_from, 
    t_point3 look_at,
    float defocus_angle,
    float focus_distance
) {
    t_camera cam;

    cam.image_width = image_width;
    cam.image_height = (int) (image_width/aspect_ratio);
    cam.image_height = (cam.image_height < 1.0F) ? 1.0F : cam.image_height;
    
    cam.center = look_from;

    // pixel_delta_u, pixel_delta_v
    t_vec3 w = vec3_unit(subtract(look_from, look_at));
    t_vec3 u = vec3_unit(cross((t_vec3) UP_DIRECTION, w));
    t_vec3 v = cross(w, u);
    float viewport_height = \
        2.0F * tan(degrees_to_radians(vertical_fov)/2.0F) * focus_distance;
    float viewport_width = viewport_height * image_width/cam.image_height;
    t_vec3 viewport_u = scale(u, viewport_width);
    t_vec3 viewport_v = scale(v, -viewport_height);
    cam.pixel_delta_u = divide(viewport_u, cam.image_width);
    cam.pixel_delta_v = divide(viewport_v, cam.image_height);

    cam.defocus_angle_is_negative = (defocus_angle <= 0.0F);

    cam.pixel00 = cam.center;
    cam.pixel00 = subtract(cam.pixel00, scale(w, focus_distance));
    cam.pixel00 = subtract(cam.pixel00, divide(viewport_u, 2.0F));
    cam.pixel00 = subtract(cam.pixel00, divide(viewport_v, 2.0F));
    cam.pixel00 = sum(cam.pixel00, scale(
        sum(cam.pixel_delta_u, cam.pixel_delta_v), 0.5F));

    // defocus_disk_u, defocus_disk_v
    float defocus_radius = \
        focus_distance * tan(degrees_to_radians(defocus_angle / 2.0F));
    cam.defocus_disk_u = scale(u, defocus_radius);
    cam.defocus_disk_v = scale(v, defocus_radius);

    return cam;
}

// Return a random ray with the end in a random point inside the (i, j) pixel.
__host__ void h_get_random_ray(t_ray *r, t_camera *cam, int i, int j) {
    // Offset from the center of the vector generated in the unit square
    // [-0.5, 0.5]x[-0.5, 0.5]
    t_vec3 offset  = vec3_new(
        h_random_float() - 0.5F, 
        h_random_float() - 0.5F, 
        0.0
    );

    // Use the offset to select a random point inside the (i, j) pixel
    t_point3 pixel_sample = cam->pixel00;
    pixel_sample = sum(pixel_sample, scale(cam->pixel_delta_u, i+offset.x));
    pixel_sample = sum(pixel_sample, scale(cam->pixel_delta_v, j+offset.y));

    t_point3 p = h_random_in_unit_disk();
    t_point3 defocus_disk_sample = cam->center;
    defocus_disk_sample = \
        sum(defocus_disk_sample, scale(cam->defocus_disk_u, p.x));
    defocus_disk_sample = \
        sum(defocus_disk_sample, scale(cam->defocus_disk_v, p.y));
    t_point3 ray_origin = (cam->defocus_angle_is_negative) ? 
                            cam->center : defocus_disk_sample;
    t_vec3 ray_direction = subtract(pixel_sample, ray_origin);

    *r = ray_new(ray_origin, ray_direction);
}
__device__ void d_get_random_ray(
    t_ray *r, 
    t_camera *cam, 
    int i, 
    int j, 
    curandState *state
) {

    t_vec3 offset = vec3_new(
        d_random_float(state) - 0.5F,
        d_random_float(state) - 0.5F,
        0.0F
    );

    t_point3 pixel_sample = sum(
        cam->pixel00,
        scale(cam->pixel_delta_u, i + offset.x)
    );
    pixel_sample = sum(
        pixel_sample,
        scale(cam->pixel_delta_v, j + offset.y)
    );

    t_point3 p = d_random_in_unit_disk(state);
    t_point3 defocus_disk_sample = sum(
        cam->center,
        scale(cam->defocus_disk_u, p.x)
    );
    defocus_disk_sample = sum(
        defocus_disk_sample,
        scale(cam->defocus_disk_v, p.y)
    );

    t_point3 ray_origin = cam->defocus_angle_is_negative ? 
        cam->center : 
        defocus_disk_sample;

    t_vec3 ray_direction = subtract(
        pixel_sample,
        ray_origin
    );

    *r = ray_new(ray_origin, ray_direction);
}


// Determining a different color for each of the pixels of the viewport by 
// sending one or more rays from the camera center to each pixel 
__host__ void h_ray_color(
    t_color *color, 
    t_ray *r, 
    t_sphere world[],
    int number_of_spheres, 
    int *bounces
) { 
    bool hit_anything = false;

    // Limit the amount of recursive calls
    if (*bounces == 0) {
        *color = COLOR_BLACK;
        return;
    }

    t_hit_result temp, result;
    float closest_hit = RAY_T_MAX;

    // For each sphere, if the ray hits the sphere before all the other spheres
    // (0.001 lower bound is used to fix "shadow acne")
    for (int i = 0; i < number_of_spheres; i++) {
        h_sphere_hit(&temp, r, world[i], 0.001F, closest_hit);
        if (temp.did_hit) {
            hit_anything = true;
            closest_hit = temp.t;
            result = temp;
        } 
    }

    if (hit_anything) {
        // Calculate color and attenuation of the scatered ray
        t_color attenuation;
        t_ray scattered_ray;
        h_scatter(&result, r, &attenuation, &scattered_ray);

        t_color scattered_ray_color;
        int next_bounces = (*bounces)-1;
        h_ray_color(&scattered_ray_color, &scattered_ray, world, 
                        number_of_spheres, &next_bounces);
        *color = mul(attenuation, scattered_ray_color);
        return;
    }

    // If no object is hit, return a blend between blue and white based on the 
    // y coordinate, so going vertically from white all the way to blue
    *color = BLEND(vec3_unit(r->direction).y, COLOR_WHITE, COLOR_SKY);
}

__device__ void d_ray_color(
    t_color *color, 
    t_ray *r, 
    t_sphere world[], 
    int number_of_spheres, 
    int *bounces,
    curandState *state
) {

    *color = COLOR_WHITE;
    int bounces_remaining = *bounces;
    t_color attenuation;
    t_ray scattered_ray;

    while (bounces_remaining > 0) {
        bool hit_anything = false;
        float closest_hit = RAY_T_MAX;
        t_hit_result result;

        // For each sphere, if the ray hits the sphere before all the other 
        // spheres (0.001 lower bound is used to fix "shadow acne")
        for (int i = 0; i < number_of_spheres; i++) {
            d_sphere_hit(&result, r, world[i], 0.001F, closest_hit);
            if (result.did_hit) {
                hit_anything = true;
                closest_hit = result.t;
            }
        }

        // If no object is hit, return a blend between blue and white base
        //  on the y coordinate, so going vertically from white all the way
        //  to blue
        if (!hit_anything) {
            *color = mul(
                *color, 
                BLEND(
                    vec3_unit(r->direction).y,
                    COLOR_WHITE,
                    COLOR_SKY
                )
            );
            break;
        }

        d_scatter(
            &result, 
            r, 
            &attenuation, 
            &scattered_ray, 
            state
        );

        // Calculate color and attenuation of the scattered ray
        *color = mul(*color, attenuation);
        *r = scattered_ray;
        bounces_remaining--;
    }
}


void h_camera_render(t_camera *cam, t_sphere world[], int number_of_spheres, 
    unsigned char *result_img) {

    // Render cycle
    int max_ray_bounces = MAX_RAY_BOUNCES;
    for (int j = 0; j < cam->image_height; j++) {
        for (int i = 0; i < cam->image_width; i++) {
            long rgb_offset = (j*(cam->image_width) + i)*3;

            t_color pixel_color = color_new(0.0F, 0.0F, 0.0F);

            // Antialiasing: sample SAMPLE_PER_PIXEL colors and average them to
            // obtain pixel color
            for (int sample = 0; sample < SAMPLES_PER_PIXEL; sample++) {
                t_ray random_ray;
                h_get_random_ray(&random_ray, cam, i, j);
                t_color sampled_color;
                h_ray_color(&sampled_color, &random_ray, world, 
                    number_of_spheres, &max_ray_bounces);
                pixel_color = sum(pixel_color, sampled_color);
            } 
            pixel_color = divide(pixel_color, SAMPLES_PER_PIXEL);

            color_write_at(pixel_color, rgb_offset, result_img);
        }
        // Update progress counter
        fprintf(stderr, 
            "\rScanlines processed: %d/%d", j + 1, cam->image_height);
        fflush(stderr);
    }

    fprintf(stderr, "\rDone.                                    \n");
}

__global__ void d_camera_render(
    t_camera *cam, 
    t_sphere world[], 
    int number_of_spheres,
    unsigned char *result_img,
    curandState random_states[]
) {
    // int thread_idx = blockDim.x * threadIdx.y + threadIdx.x;
    int pixel_i = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((pixel_i < cam->image_width) && (pixel_j < cam->image_height)) {
        long pixel_index = pixel_j * (cam->image_width) + pixel_i;
        long rgb_offset = pixel_index * 3;

        t_color pixel_color = color_new(0.0F, 0.0F, 0.0F);
        t_color sampled_color;
        t_ray random_ray;
        int max_ray_bounces = MAX_RAY_BOUNCES;
        long linear_index = (blockIdx.y*gridDim.x + blockIdx.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x + threadIdx.x);
        for (int sample = 0; sample < SAMPLES_PER_PIXEL; sample++) {
            d_get_random_ray(&random_ray, cam, pixel_i, pixel_j, random_states + linear_index);
            d_ray_color(&sampled_color, &random_ray, world, NUMBER_OF_SPHERES, &max_ray_bounces, random_states + linear_index);
            pixel_color = sum(pixel_color, sampled_color);
        }
        sampled_color = divide(pixel_color, SAMPLES_PER_PIXEL);

        color_write_at(sampled_color, rgb_offset, result_img);
    }
}

#endif