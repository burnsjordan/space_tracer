#![feature(test)]

extern crate test;
use test::{black_box, Bencher};
use space_tracer::*;

#[bench]
fn bench_intersect_ray_sphere(b: &mut Bencher) {
    // Scene setup
    let mut window_width = space_tracer::STARTING_SCREEN_WIDTH;
    let mut window_height = space_tracer::STARTING_SCREEN_HEIGHT;
    let camera_pos = space_tracer::CAMERA_STARTING_POSITION;
    let camera_rot = Vec3 {
        a: space_tracer::CAMERA_STARTING_ROTATION_X_ANGLE,
        b: space_tracer::CAMERA_STARTING_ROTATION_Y_ANGLE,
        c: space_tracer::CAMERA_STARTING_ROTATION_Z_ANGLE,
    };
    let camera_rotation_matrix: RotationMatrix3D =
        build_rotation_matrix_3d(camera_rot.a, camera_rot.b, camera_rot.c);
    let scene_spheres = build_scene_objects();

    if window_width % 2 != 0 {
        window_width += 1;
    }
    if window_height % 2 != 0 {
        window_height += 1;
    }

    let ray_direction: Vec3 = camera_rotation_matrix
                                    * canvas_to_viewport(
                                        0.0,
                                        0.0,
                                        window_width,
                                        window_height,
                                    );
    let a: f32 = dot_product(ray_direction, ray_direction);

    b.iter(|| {
        for _i in 1..100 {
            black_box(space_tracer::intersect_ray_sphere(camera_pos, ray_direction, scene_spheres[0], a));
        }
    });
}

#[bench]
fn bench_closest_intersection(b: &mut Bencher) {
    // Scene setup
    let mut window_width = space_tracer::STARTING_SCREEN_WIDTH;
    let mut window_height = space_tracer::STARTING_SCREEN_HEIGHT;
    let camera_pos = space_tracer::CAMERA_STARTING_POSITION;
    let camera_rot = Vec3 {
        a: space_tracer::CAMERA_STARTING_ROTATION_X_ANGLE,
        b: space_tracer::CAMERA_STARTING_ROTATION_Y_ANGLE,
        c: space_tracer::CAMERA_STARTING_ROTATION_Z_ANGLE,
    };
    let camera_rotation_matrix: RotationMatrix3D =
        build_rotation_matrix_3d(camera_rot.a, camera_rot.b, camera_rot.c);
    let scene_spheres = build_scene_objects();

    if window_width % 2 != 0 {
        window_width += 1;
    }
    if window_height % 2 != 0 {
        window_height += 1;
    }

    let ray_direction: Vec3 = camera_rotation_matrix
                                    * canvas_to_viewport(
                                        0.0,
                                        0.0,
                                        window_width,
                                        window_height,
                                    );

    b.iter(|| {
        for _i in 1..100 {
            black_box(space_tracer::closest_intersection(camera_pos, ray_direction, space_tracer::EPSILON, space_tracer::INF, &scene_spheres));
        }
    });
}

#[bench]
fn bench_closest_intersection_shadow(b: &mut Bencher) {
    // Scene setup
    let mut window_width = space_tracer::STARTING_SCREEN_WIDTH;
    let mut window_height = space_tracer::STARTING_SCREEN_HEIGHT;
    let camera_pos = space_tracer::CAMERA_STARTING_POSITION;
    let camera_rot = Vec3 {
        a: space_tracer::CAMERA_STARTING_ROTATION_X_ANGLE,
        b: space_tracer::CAMERA_STARTING_ROTATION_Y_ANGLE,
        c: space_tracer::CAMERA_STARTING_ROTATION_Z_ANGLE,
    };
    let camera_rotation_matrix: RotationMatrix3D =
        build_rotation_matrix_3d(camera_rot.a, camera_rot.b, camera_rot.c);
    let scene_spheres = build_scene_objects();

    if window_width % 2 != 0 {
        window_width += 1;
    }
    if window_height % 2 != 0 {
        window_height += 1;
    }

    let ray_direction: Vec3 = camera_rotation_matrix
                                    * canvas_to_viewport(
                                        0.0,
                                        0.0,
                                        window_width,
                                        window_height,
                                    );

    b.iter(|| {
        for _i in 1..100 {
            black_box(space_tracer::closest_intersection_shadow(camera_pos, ray_direction, space_tracer::EPSILON, space_tracer::INF, &scene_spheres));
        }
    });
}

#[bench]
fn bench_compute_lighting(b: &mut Bencher) {
    // Scene setup
    let mut window_width = space_tracer::STARTING_SCREEN_WIDTH;
    let mut window_height = space_tracer::STARTING_SCREEN_HEIGHT;
    let camera_pos = space_tracer::CAMERA_STARTING_POSITION;
    let camera_rot = Vec3 {
        a: space_tracer::CAMERA_STARTING_ROTATION_X_ANGLE,
        b: space_tracer::CAMERA_STARTING_ROTATION_Y_ANGLE,
        c: space_tracer::CAMERA_STARTING_ROTATION_Z_ANGLE,
    };
    let camera_rotation_matrix: RotationMatrix3D =
        build_rotation_matrix_3d(camera_rot.a, camera_rot.b, camera_rot.c);
    let scene_spheres = build_scene_objects();
    let scene_lights = build_scene_lights();

    if window_width % 2 != 0 {
        window_width += 1;
    }
    if window_height % 2 != 0 {
        window_height += 1;
    }

    let ray_direction: Vec3 = camera_rotation_matrix
                                    * canvas_to_viewport(
                                        0.0,
                                        0.0,
                                        window_width,
                                        window_height,
                                    );

    let p: Vec3 = camera_pos + (ray_direction * 1000.0);
    let n: Vec3 = p;
    let n = n / vector_length(n);

    b.iter(|| {
        for _i in 1..100 {
            black_box(space_tracer::compute_lighting(p, n, -1.0 * ray_direction, &scene_spheres, &scene_lights, 1000.0));
        }
    });
}

#[bench]
fn bench_trace_ray(b: &mut Bencher) {
    // Scene setup
    let mut window_width = space_tracer::STARTING_SCREEN_WIDTH;
    let mut window_height = space_tracer::STARTING_SCREEN_HEIGHT;
    let camera_pos = space_tracer::CAMERA_STARTING_POSITION;
    let camera_rot = Vec3 {
        a: space_tracer::CAMERA_STARTING_ROTATION_X_ANGLE,
        b: space_tracer::CAMERA_STARTING_ROTATION_Y_ANGLE,
        c: space_tracer::CAMERA_STARTING_ROTATION_Z_ANGLE,
    };
    let camera_rotation_matrix: RotationMatrix3D =
        build_rotation_matrix_3d(camera_rot.a, camera_rot.b, camera_rot.c);
    let scene_spheres = build_scene_objects();
    let scene_lights = build_scene_lights();

    if window_width % 2 != 0 {
        window_width += 1;
    }
    if window_height % 2 != 0 {
        window_height += 1;
    }

    let ray_direction: Vec3 = camera_rotation_matrix
                                    * canvas_to_viewport(
                                        0.0,
                                        0.0,
                                        window_width,
                                        window_height,
                                    );

    b.iter(|| {
        for _i in 1..100 {
            black_box(space_tracer::trace_ray(camera_pos, ray_direction, space_tracer::EPSILON, space_tracer::INF, &scene_spheres, &scene_lights, space_tracer::REFLECTION_RECURSION_DEPTH));
        }
    });
}

#[ignore]
#[bench]
fn bench_render_scene(b: &mut Bencher) {
    // Scene setup
    let mut window_width = space_tracer::STARTING_SCREEN_WIDTH;
    let mut window_height = space_tracer::STARTING_SCREEN_HEIGHT;
    let camera_pos = space_tracer::CAMERA_STARTING_POSITION;
    let camera_rot = Vec3 {
        a: space_tracer::CAMERA_STARTING_ROTATION_X_ANGLE,
        b: space_tracer::CAMERA_STARTING_ROTATION_Y_ANGLE,
        c: space_tracer::CAMERA_STARTING_ROTATION_Z_ANGLE,
    };
    let camera_rotation_matrix: RotationMatrix3D =
        build_rotation_matrix_3d(camera_rot.a, camera_rot.b, camera_rot.c);
    let scene_spheres = build_scene_objects();
    let scene_lights = build_scene_lights();

    if window_width % 2 != 0 {
        window_width += 1;
    }
    if window_height % 2 != 0 {
        window_height += 1;
    }

    b.iter(|| {
        for _i in 1..2 {
            black_box(space_tracer::render_scene(
                &scene_spheres,
                &scene_lights,
                camera_pos,
                camera_rotation_matrix,
                window_width,
                window_height,
            ));
        }
    });
}
