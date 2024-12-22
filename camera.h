#ifndef CAMERA_H
#define CAMERA_H

#include "color.h"
#include "common.h"
#include "point3.h"
#include "ray.h"
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
    t_point3 center;

    // The viewport is a virtual rectangle in our 3D world containing the 
    // square pixels in our rendered image.
    // One or more rays are sent from the camera center through each pixel of 
    // the viewport and the intersection point with the objects in the scene is 
    // determined, which in turn determines the pixel's color.
    my_decimal viewport_height, viewport_width;

    // Convenient vectors to navigate the viewport through its width and down 
    // its height.
    t_vec3 viewport_u, viewport_v;
    t_vec3 pixel_delta_u, pixel_delta_v;

    // Center of upper-left pixel in the viewpoer
    t_point3 pixel00;

    // Vertical field of view, aka the angle between the image top and bottom.
    // Expressed in degrees
    my_decimal vertical_fov; 

    // Where the camera center from (camera center) and where it looks towards
    t_point3 look_from, look_at;

    // What the camera considers to be "up"
    t_vec3 up_direction;

    my_decimal defocus_angle;

    // How distant a plane with perfect focus is from the lookfrom angle
    my_decimal focus_dist;

    // Horizontal and vertical radius of defocus disk
    t_vec3 defocus_disk_u, defocus_disk_v;
} t_camera;

// Constructor
t_camera camera_new(my_decimal aspect_ratio, 
                    int image_width, 
                    my_decimal fov, 
                    t_point3 look_from, 
                    t_point3 look_at,
                    my_decimal defocus_angle,
                    my_decimal focus_dist) {
    t_camera cam;

    cam.image_height = (int) (image_width/aspect_ratio);
    cam.image_height = (cam.image_height < 1) ? 1 : cam.image_height;
    cam.image_width = image_width;

    cam.look_from = look_from;
    cam.look_at = look_at;
    cam.up_direction = vec3_new(0, 1, 0);

    cam.center = look_from;

    t_vec3 w = vec3_unit(subtract(look_from, look_at));
    t_vec3 u = vec3_unit(cross(cam.up_direction, w));
    t_vec3 v = cross(w, u);

    cam.focus_dist = 10;
    cam.defocus_angle = 0;

    cam.viewport_height = \
        2.0 * tan(degrees_to_radians(fov)/2.0) * cam.focus_dist;
    cam.viewport_width = cam.viewport_height * 
        (((my_decimal) image_width)/cam.image_height);
    cam.viewport_u = scale(u, cam.viewport_width);
    cam.viewport_v = scale(v, -cam.viewport_height);
    cam.pixel_delta_u = divide(cam.viewport_u, cam.image_width);
    cam.pixel_delta_v = divide(cam.viewport_v, cam.image_height);

    cam.pixel00 = cam.center;
    cam.pixel00 = subtract(cam.pixel00, scale(w, cam.focus_dist));
    cam.pixel00 = subtract(cam.pixel00, divide(cam.viewport_u, 2));
    cam.pixel00 = subtract(cam.pixel00, divide(cam.viewport_v, 2));
    cam.pixel00 = sum(cam.pixel00, scale(
        sum(cam.pixel_delta_u, cam.pixel_delta_v), 0.5));

    my_decimal defocus_radius = cam.focus_dist * tan(degrees_to_radians(cam.defocus_angle / 2));
    cam.defocus_disk_u = scale(u, defocus_radius);
    cam.defocus_disk_v = scale(v, defocus_radius);

    return cam;
}

// Return a random ray with the end in a random point inside the (i, j) pixel.
t_ray get_random_ray(t_camera *cam, int i, int j) {
    // Offset from the center of the vector generated in the unit square
    // [-0.5, 0.5]x[-0.5, 0.5]
    t_vec3 offset  = vec3_new(random_my_decimal() - 0.5, 
                            random_my_decimal() - 0.5, 0.0);

    // Use the offset to select a random point inside the (i, j) pixel
    t_point3 pixel_sample = cam->pixel00;
    pixel_sample = sum(pixel_sample, scale(cam->pixel_delta_u, i+offset.x));
    pixel_sample = sum(pixel_sample, scale(cam->pixel_delta_v, j+offset.y));

    t_vec3 ray_direction = subtract(pixel_sample, cam->center);

    t_point3 p = random_in_unit_disk();
    t_point3 defocus_disk_sample = cam->center;
    defocus_disk_sample = sum(defocus_disk_sample, scale(cam->defocus_disk_u, p.x));
    defocus_disk_sample = sum(defocus_disk_sample, scale(cam->defocus_disk_v, p.y));
    t_point3 ray_origin = (cam->defocus_angle <= 0) ? cam->center : defocus_disk_sample;
    return ray_new(cam->center, ray_direction);
}

// Determining a different color for each of the pixels of the viewport by 
// sending one or more rays from the camera center to each pixel 
t_color ray_color(t_ray *r, t_sphere world[], int number_of_spheres, 
                  int bounces, bool *hit_anything) { 
    *hit_anything = false;

    // Limit the amount of recursive calls
    if (bounces == 0) {
        return COLOR_BLACK;
    }

    t_hit_result temp, result;
    my_decimal closest_hit = RAY_T_MAX;

    // For each sphere, if the ray hits the sphere before all the other spheres
    // (0.001 lower bound is used to fix "shadow acne")
    for (int i = 0; i < number_of_spheres; i++) {
        temp = sphere_hit(r, world[i], 0.001, closest_hit);
        if (temp.did_hit) {
            *hit_anything = true;
            closest_hit = temp.t;
            result = temp;
        } 
    }

    if (*hit_anything) {
        t_color attenuation;
        t_ray scattered_ray;

        // Return the color of the scattered ray
        if ( scatter(&result, r, &attenuation, &scattered_ray) ) {
            t_color scattered_ray_color = ray_color(&scattered_ray, world, 
                                   number_of_spheres, bounces-1, hit_anything);
            return mul(attenuation, scattered_ray_color);
        }

        return COLOR_BLACK;
    }

    // If no object is hit, return a blend between blue and white based on the 
    // y coordinate, so going vertically from white all the way to blue
    return BLEND(vec3_unit(r->direction).y, COLOR_WHITE, COLOR_SKY);
}

void camera_render(t_camera *cam, t_sphere world[], int number_of_spheres) {
    // PPM header
    printf("P3\n%d %d\n255\n", cam->image_width, cam->image_height);

    // Render cycle
    bool dummy = false;
    for (int j = 0; j < cam->image_height; j++) {
        for (int i = 0; i < cam->image_width; i++) {
            t_color pixel_color = color_new(0, 0, 0);

            // Antialiasing: sample SAMPLE_PER_PIXEL colors and average them to 
            // obtain pixel color
            for (int sample = 0; sample < SAMPLES_PER_PIXEL; sample++) {
                t_ray random_ray = get_random_ray(cam, i, j);
                t_color sampled_color = ray_color(&random_ray, world, 
                    number_of_spheres, MAX_RAY_BOUNCES, &dummy);
                pixel_color = sum(pixel_color, sampled_color);
            } 
            pixel_color = divide(pixel_color, SAMPLES_PER_PIXEL);

            color_print_PPM(pixel_color);
        }
        // Update progress counter
        fprintf(stderr, "\rScanlines processed: %d/%d", j + 1, cam->image_height);
        fflush(stderr);
    }

    fprintf(stderr, "\rDone.                                    \n");
}

#endif