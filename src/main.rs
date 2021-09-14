extern crate gl;
extern crate glutin;

use gl::types::*;
use std::{ffi::CString, mem, ops, ptr, str, sync::mpsc, thread, time, time::SystemTime};

// Settings
const STARTING_SCREEN_HEIGHT: i32 = 768;
const STARTING_SCREEN_WIDTH: i32 = 768;
const BUFFER_DATA_LENGTH: i32 = 5;
const SCREEN_MULTIPLIER: i32 = 1;
const SIZE_OF_FLOAT: i32 = 4;
const INF: f32 = 99999999.0;
const EPSILON: f32 = 0.001;
const REFLECTION_RECURSION_DEPTH: u8 = 0;
const NUM_THREADS: u8 = 6; // Needs to be even or one
const SHOW_DEBUG_LOGS: bool = true;
const SUBSAMP_RATE: u8 = 1; // 1 disables subsampling
const SHADOWS_ON: bool = true;
const RENDER_STATIC_IMAGE: bool = false; // Disables re-rendering scene
const START_FULLSCREEN: bool = false;
const START_MAXIMIZED: bool = false;
const MAX_FPS: u8 = 60;
const MAX_FRAME_TIME_SEC: f32 = 1.0 / MAX_FPS as f32;

// Viewport Settings
const DISTANCE_TO_VIEWPORT: f32 = 1.0;

// Camera Settings
const CAMERA_STARTING_POSITION: Vec3 = Vec3 {
    a: 0.0,
    b: 1.0,
    c: 0.0,
};
const CAMERA_STARTING_ROTATION_X_ANGLE: f32 = 0.0;
const CAMERA_STARTING_ROTATION_Y_ANGLE: f32 = 0.0;
const CAMERA_STARTING_ROTATION_Z_ANGLE: f32 = 0.5;
const CAMERA_ROTATION_SPEED: f32 = 5.0;
const CAMERA_MOVEMENT_SPEED: f32 = 5.0;

#[derive(Copy, Clone)]
struct Vec3 {
    a: f32,
    b: f32,
    c: f32,
}

impl ops::Add for Vec3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Vec3 {
            a: self.a + other.a,
            b: self.b + other.b,
            c: self.c + other.c,
        }
    }
}

impl ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Vec3 {
            a: self.a - other.a,
            b: self.b - other.b,
            c: self.c - other.c,
        }
    }
}

impl ops::Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, other: f32) -> Self {
        Vec3 {
            a: self.a * other,
            b: self.b * other,
            c: self.c * other,
        }
    }
}

impl ops::Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, other: Vec3) -> Vec3 {
        Vec3 {
            a: self * other.a,
            b: self * other.b,
            c: self * other.c,
        }
    }
}

impl ops::Div<f32> for Vec3 {
    type Output = Self;

    fn div(self, other: f32) -> Self {
        Vec3 {
            a: self.a / other,
            b: self.b / other,
            c: self.c / other,
        }
    }
}

#[derive(Copy, Clone)]
struct RotationMatrix3D {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    h: f32,
    i: f32,
}

impl ops::Mul<Vec3> for RotationMatrix3D {
    type Output = Vec3;

    fn mul(self, other: Vec3) -> Vec3 {
        Vec3 {
            a: self.a * other.a + self.b * other.b + self.c * other.c,
            b: self.d * other.a + self.e * other.b + self.f * other.c,
            c: self.g * other.a + self.h * other.b + self.i * other.c,
        }
    }
}

fn build_rotation_matrix_3d(alpha: f32, beta: f32, gamma: f32) -> RotationMatrix3D {
    RotationMatrix3D {
        a: alpha.cos() * beta.cos(),
        b: alpha.cos() * beta.sin() * gamma.sin() - alpha.sin() * gamma.cos(),
        c: alpha.cos() * beta.sin() * gamma.cos() + alpha.sin() * gamma.sin(),
        d: alpha.sin() * beta.cos(),
        e: alpha.sin() * beta.sin() * gamma.sin() + alpha.cos() * gamma.cos(),
        f: alpha.sin() * beta.sin() * gamma.cos() - alpha.cos() * gamma.sin(),
        g: -1.0 * beta.sin(),
        h: beta.cos() * gamma.sin(),
        i: beta.cos() * gamma.cos(),
    }
}

#[derive(Copy, Clone)]
struct Sphere {
    center: Vec3,
    radius: f32,
    color: (u8, u8, u8),
    specular: f32,
    reflective: f32,
}

#[derive(Clone)]
enum LightType {
    Ambient,
    Point,
    Directional,
}

#[derive(Clone)]
struct Light {
    light_type: LightType,
    intensity: f32,
    position: Option<Vec3>,
    direction: Option<Vec3>,
}

// Shader sources
static VS_SRC: &str = "
#version 330 core
layout (location = 0) in vec2 position;
layout (location = 1) in vec3 color;
out vec4 vertexColor;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    vertexColor = vec4(color, 1.0);
}";

static FS_SRC: &str = "
#version 330 core
out vec4 out_color;
in vec4 vertexColor;
void main() {
    out_color = vertexColor;
}";

fn compile_shader(src: &str, ty: GLenum) -> GLuint {
    let shader;
    unsafe {
        shader = gl::CreateShader(ty);
        // Attempt to compile the shader
        let c_str = CString::new(src.as_bytes()).unwrap();
        gl::ShaderSource(shader, 1, &c_str.as_ptr(), ptr::null());
        gl::CompileShader(shader);

        // Get the compile status
        let mut status = gl::FALSE as GLint;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut status);

        // Fail on error
        if status != (gl::TRUE as GLint) {
            let mut len = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = Vec::with_capacity(len as usize);
            buf.set_len((len as usize) - 1); // subtract 1 to skip the trailing null character
            gl::GetShaderInfoLog(
                shader,
                len,
                ptr::null_mut(),
                buf.as_mut_ptr() as *mut GLchar,
            );
            panic!(
                "{}",
                str::from_utf8(&buf).expect("ShaderInfoLog not valid utf8")
            );
        }
    }
    shader
}

fn link_program(vs: GLuint, fs: GLuint) -> GLuint {
    unsafe {
        let program = gl::CreateProgram();
        gl::AttachShader(program, vs);
        gl::AttachShader(program, fs);
        gl::LinkProgram(program);
        // Get the link status
        let mut status = gl::FALSE as GLint;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut status);

        // Fail on error
        if status != (gl::TRUE as GLint) {
            let mut len: GLint = 0;
            gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = Vec::with_capacity(len as usize);
            buf.set_len((len as usize) - 1); // subtract 1 to skip the trailing null character
            gl::GetProgramInfoLog(
                program,
                len,
                ptr::null_mut(),
                buf.as_mut_ptr() as *mut GLchar,
            );
            panic!(
                "{}",
                str::from_utf8(&buf).expect("ProgramInfoLog not valid utf8")
            );
        }
        program
    }
}

fn min(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}

fn setup_graphics(program: u32) -> (u32, u32) {
    let mut vao = 0;
    let mut vbo = 0;
    unsafe {
        // Create Vertex Array Object
        gl::GenVertexArrays(1, &mut vao);
        gl::BindVertexArray(vao);

        // Create a Vertex Buffer Object and copy the vertex data to it
        gl::GenBuffers(1, &mut vbo);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::UseProgram(program);
        let out_color_c_string = CString::new("out_color").unwrap();
        gl::BindFragDataLocation(program, 0, out_color_c_string.as_ptr());
        let position_c_string = CString::new("position").unwrap();
        let position = gl::GetAttribLocation(program, position_c_string.as_ptr());
        gl::EnableVertexAttribArray(position as GLuint);
        gl::VertexAttribPointer(
            position as GLuint,
            2,
            gl::FLOAT,
            gl::FALSE as GLboolean,
            5 * SIZE_OF_FLOAT,
            ptr::null(),
        );
        let color_c_string = CString::new("color").unwrap();
        let color = gl::GetAttribLocation(program, color_c_string.as_ptr());
        gl::EnableVertexAttribArray(color as GLuint);
        gl::VertexAttribPointer(
            color as GLuint,
            3,
            gl::FLOAT,
            gl::FALSE as GLboolean,
            5 * SIZE_OF_FLOAT,
            (2 * SIZE_OF_FLOAT) as *const std::ffi::c_void,
        );
        gl::PointSize(SCREEN_MULTIPLIER as f32);
    }
    (vao, vbo)
}

fn draw_graphics(
    gl_window: &glutin::ContextWrapper<glutin::PossiblyCurrent, glutin::window::Window>,
    vertex_data: &[GLfloat],
) {
    unsafe {
        // Clear the screen to black
        gl::ClearColor(0.0, 0.0, 0.0, 0.0);
        gl::Clear(gl::COLOR_BUFFER_BIT);
        if !vertex_data.is_empty() {
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (vertex_data.len() * mem::size_of::<GLfloat>()) as GLsizeiptr,
                mem::transmute(&vertex_data[0]),
                gl::STATIC_DRAW,
            );
            // Draw pixels from vertex data
            gl::DrawArrays(gl::POINTS, 0, vertex_data.len() as i32 / BUFFER_DATA_LENGTH);
            gl::Viewport(
                0,
                0,
                gl_window.window().inner_size().width as i32,
                gl_window.window().inner_size().height as i32,
            );
        }
    }
    gl_window.swap_buffers().unwrap();
}

fn put_pixel(x: f32, y: f32, color: (u8, u8, u8), vertex_data: &mut Vec<GLfloat>) {
    // Center 0 coordinates
    vertex_data.push(x as GLfloat);
    vertex_data.push(y as GLfloat);
    vertex_data.push(color.0 as f32 / 255.0);
    vertex_data.push(color.1 as f32 / 255.0);
    vertex_data.push(color.2 as f32 / 255.0);
}

fn canvas_to_viewport(cx: f32, cy: f32, screen_width: i32, screen_height: i32) -> Vec3 {
    Vec3 {
        a: cx / screen_width as f32,
        b: cy * (screen_height as f32 / screen_width as f32) / screen_height as f32,
        c: DISTANCE_TO_VIEWPORT,
    }
}

fn dot_product(a: Vec3, b: Vec3) -> f32 {
    let mut product = 0.0;
    product += a.a * b.a;
    product += a.b * b.b;
    product += a.c * b.c;
    product
}

fn vector_length(a: Vec3) -> f32 {
    let mut sum = 0.0;
    sum += a.a * a.a;
    sum += a.b * a.b;
    sum += a.c * a.c;
    sum.sqrt()
}

fn intersect_ray_sphere(
    camera_pos: Vec3,
    ray_direction: Vec3,
    sphere: Sphere,
    a: f32,
) -> (f32, f32) {
    let r: f32 = sphere.radius;
    let co: Vec3 = camera_pos - sphere.center;

    let b: f32 = 2.0 * dot_product(co, ray_direction);
    let c: f32 = dot_product(co, co) - (r * r);

    let discriminant: f32 = (b * b) - (4.0 * a * c);
    if discriminant < 0.0 {
        (INF, INF)
    } else {
        let t1: f32 = (-b + discriminant.sqrt()) / (2.0 * a);
        let t2: f32 = (-b - discriminant.sqrt()) / (2.0 * a);
        (t1, t2)
    }
}

fn closest_intersection(
    camera_pos: Vec3,
    ray_direction: Vec3,
    t_min: f32,
    t_max: f32,
    spheres: &[Sphere],
) -> (Option<Sphere>, f32) {
    let mut closest_t: f32 = INF;
    let mut closest_sphere: Option<Sphere> = None;
    let a: f32 = dot_product(ray_direction, ray_direction);
    for sphere in spheres.iter() {
        let ips = intersect_ray_sphere(camera_pos, ray_direction, *sphere, a);
        if ips.0 < t_max && ips.0 > t_min && ips.0 < closest_t {
            closest_t = ips.0;
            closest_sphere = Some(*sphere);
        }
        if ips.1 < t_max && ips.1 > t_min && ips.1 < closest_t {
            closest_t = ips.1;
            closest_sphere = Some(*sphere);
        }
    }
    (closest_sphere, closest_t)
}

// For shadows it doesn't need to be the closest intersection
// just any intersection so return when any is found
fn closest_intersection_shadow(
    camera_pos: Vec3,
    ray_direction: Vec3,
    t_min: f32,
    t_max: f32,
    spheres: &[Sphere],
) -> Option<Sphere> {
    let closest_t: f32 = INF;
    let mut closest_sphere: Option<Sphere> = None;
    let a: f32 = dot_product(ray_direction, ray_direction);
    for sphere in spheres.iter() {
        let ips = intersect_ray_sphere(camera_pos, ray_direction, *sphere, a);
        if ips.0 < t_max && ips.0 > t_min && ips.0 < closest_t {
            closest_sphere = Some(*sphere);
            return closest_sphere;
        }
        if ips.1 < t_max && ips.1 > t_min && ips.1 < closest_t {
            closest_sphere = Some(*sphere);
            return closest_sphere;
        }
    }
    closest_sphere
}

fn reflect_ray(r: Vec3, n: Vec3) -> Vec3 {
    (2.0 * n * dot_product(n, r)) - r
}

fn compute_lighting(
    point_position: Vec3,
    surface_normal: Vec3,
    v: Vec3,
    spheres: &[Sphere],
    light: &[Light],
    specular: f32,
) -> f32 {
    let mut i = 0.0;
    let mut l: Vec3;
    let mut t_max: f32;
    for light in light.iter() {
        match light.light_type {
            LightType::Ambient => {
                i += light.intensity;
            }
            LightType::Point => {
                if light.position.is_none() {
                    l = point_position;
                } else {
                    let t: Vec3 = light
                        .position
                        .or(Some(Vec3 {
                            a: 0.0,
                            b: 0.0,
                            c: 0.0,
                        }))
                        .unwrap();
                    l = t - point_position;
                }

                t_max = 1.0;
                let shadow_sphere: Option<Sphere>;
                if !SHADOWS_ON {
                    shadow_sphere = None;
                } else {
                    shadow_sphere =
                        closest_intersection_shadow(point_position, l, EPSILON, t_max, spheres);
                }
                if shadow_sphere.is_none() {
                    let n_dot_l: f32 = dot_product(surface_normal, l);
                    if n_dot_l > 0.0 {
                        i += light.intensity * n_dot_l
                            / (vector_length(surface_normal) * vector_length(l));
                    }
                    if (specular - -1.0).abs() > f32::EPSILON {
                        let r = (2.0 * surface_normal * dot_product(surface_normal, l)) - l;
                        let r_dot_v = dot_product(r, v);
                        if r_dot_v > 0.0 {
                            i += light.intensity
                                * (r_dot_v / (vector_length(r) * vector_length(v))).powf(specular);
                        }
                    }
                }
            }
            LightType::Directional => {
                if light.direction.is_none() {
                    l = Vec3 {
                        a: 0.0,
                        b: 0.0,
                        c: 0.0,
                    };
                } else {
                    l = light
                        .direction
                        .or(Some(Vec3 {
                            a: 0.0,
                            b: 0.0,
                            c: 0.0,
                        }))
                        .unwrap();
                }

                t_max = INF;
                let shadow_sphere: Option<Sphere>;
                if !SHADOWS_ON {
                    shadow_sphere = None;
                } else {
                    shadow_sphere =
                        closest_intersection_shadow(point_position, l, EPSILON, t_max, spheres);
                }
                if shadow_sphere.is_none() {
                    let n_dot_l: f32 = dot_product(surface_normal, l);
                    if n_dot_l > 0.0 {
                        i += light.intensity * n_dot_l
                            / (vector_length(surface_normal) * vector_length(l));
                    }
                    if (specular - -1.0).abs() > f32::EPSILON {
                        let r = (2.0 * surface_normal * dot_product(surface_normal, l)) - l;
                        let r_dot_v = dot_product(r, v);
                        if r_dot_v > 0.0 {
                            i += light.intensity
                                * (r_dot_v / (vector_length(r) * vector_length(v))).powf(specular);
                        }
                    }
                }
            }
        }
    }
    i
}

fn trace_ray(
    camera_pos: Vec3,
    ray_direction: Vec3,
    t_min: f32,
    t_max: f32,
    spheres: &[Sphere],
    lights: &[Light],
    rec_depth: u8,
) -> Option<(u8, u8, u8)> {
    let (closest_sphere, closest_t) =
        closest_intersection(camera_pos, ray_direction, t_min, t_max, spheres);
    if let Some(cs) = closest_sphere {
        let p: Vec3 = camera_pos + (ray_direction * closest_t);
        let n: Vec3 = p - cs.center;
        let n = n / vector_length(n);
        let local_color: (u8, u8, u8) = (
            (cs.color.0 as f32
                * compute_lighting(p, n, -1.0 * ray_direction, spheres, lights, cs.specular))
                as u8,
            (cs.color.1 as f32
                * compute_lighting(p, n, -1.0 * ray_direction, spheres, lights, cs.specular))
                as u8,
            (cs.color.2 as f32
                * compute_lighting(p, n, -1.0 * ray_direction, spheres, lights, cs.specular))
                as u8,
        );
        let r = cs.reflective;
        if rec_depth == 0 || r <= 0.0 {
            Some(local_color)
        } else {
            let refl_ray: Vec3 = reflect_ray(-1.0 * ray_direction, n);
            let reflected_sphere =
                trace_ray(p, refl_ray, EPSILON, INF, spheres, lights, rec_depth - 1);
            if let Some(rs) = reflected_sphere {
                Some((
                    (local_color.0 as f32 * (1.0 - r) + rs.0 as f32 * r) as u8,
                    (local_color.1 as f32 * (1.0 - r) + rs.1 as f32 * r) as u8,
                    (local_color.2 as f32 * (1.0 - r) + rs.2 as f32 * r) as u8,
                ))
            } else {
                Some(local_color)
            }
        }
    } else {
        None
    }
}

fn render_scene(
    scene_spheres: &Vec<Sphere>,
    scene_lights: &Vec<Light>,
    camera_pos: Vec3,
    camera_rotation_matrix: RotationMatrix3D,
    screen_width: i32,
    screen_height: i32,
) -> Vec<f32> {
    // Split raytracing into threads and execute
    let now = SystemTime::now();
    let mut vertex_data = vec![0.0; 0];
    let mut children = vec![];

    for i in 0..NUM_THREADS {
        let scene_spheres = scene_spheres.clone();
        let scene_lights = scene_lights.clone();
        children.push(thread::spawn(move || {
            let thread_now = SystemTime::now();
            let mut temp_vertex_data = vec![0 as f32; 0];
            let mut screen_width_per_thread = 2 * screen_width / NUM_THREADS as i32;
            let mut screen_height_per_thread = screen_height / 2;
            let xi: u8;
            let yi: u8;
            if NUM_THREADS == 1 {
                screen_height_per_thread = screen_height as i32;
                screen_width_per_thread = screen_width as i32;
                xi = 0;
                yi = 0;
            } else {
                xi = if i >= (NUM_THREADS / 2) {
                    i - (NUM_THREADS / 2)
                } else {
                    i
                };
                yi = if i >= (NUM_THREADS / 2) { 1 } else { 0 };
            }

            let left_pixel_number: i32 = (-screen_width / 2) + screen_width_per_thread * xi as i32;
            let right_pixel_number: i32 =
                (-screen_width / 2) + screen_width_per_thread * xi as i32 + screen_width_per_thread;
            let top_pixel_number: i32 = (-screen_height / 2) + screen_height_per_thread * yi as i32;
            let bottom_pixel_number: i32 = (-screen_height / 2)
                + screen_height_per_thread * yi as i32
                + screen_height_per_thread;

            for x in left_pixel_number..right_pixel_number {
                for y in top_pixel_number..bottom_pixel_number {
                    if x % SUBSAMP_RATE as i32 == 0 && y % SUBSAMP_RATE as i32 == 0 {
                        let ray_direction: Vec3 = camera_rotation_matrix
                            * canvas_to_viewport(x as f32, y as f32, screen_width, screen_height);
                        let color_option: Option<(u8, u8, u8)> = trace_ray(
                            camera_pos,
                            ray_direction,
                            1.0,
                            INF,
                            &scene_spheres,
                            &scene_lights,
                            REFLECTION_RECURSION_DEPTH,
                        );
                        if let Some(color) = color_option {
                            for i in 0..SUBSAMP_RATE {
                                for j in 0..SUBSAMP_RATE {
                                    put_pixel(
                                        (x as f32 + i as f32) / (screen_width / 2) as f32,
                                        (y as f32 + j as f32) / (screen_height / 2) as f32,
                                        color,
                                        &mut temp_vertex_data,
                                    );
                                }
                            }
                        }
                    }
                }
            }
            match thread_now.elapsed() {
                Ok(elapsed) => {
                    if SHOW_DEBUG_LOGS {
                        println!("Thread {} time: {} milliseconds", i, elapsed.as_millis());
                    }
                }
                Err(e) => {
                    println!("Error: {:?}", e);
                }
            }
            temp_vertex_data
        }));
    }

    for child in children {
        // Wait for the thread to finish. Returns a result.
        let mut temp_vertex_data = child.join().unwrap();
        vertex_data.append(&mut temp_vertex_data);
    }

    match now.elapsed() {
        Ok(elapsed) => {
            if SHOW_DEBUG_LOGS {
                println!("Total render time: {} milliseconds", elapsed.as_millis());
            }
        }
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }
    if vertex_data.is_empty() {
        put_pixel(0.0, 0.0, (0, 0, 0), &mut vertex_data);
    }
    vertex_data
}

fn build_scene_objects() -> Vec<Sphere> {
    return vec![
        Sphere {
            center: Vec3 {
                a: 0.0,
                b: -1.0,
                c: 3.0,
            },
            radius: 1.0,
            color: (255, 0, 0),
            specular: 500.0,
            reflective: 0.2,
        },
        Sphere {
            center: Vec3 {
                a: 2.0,
                b: 0.0,
                c: 4.0,
            },
            radius: 1.0,
            color: (0, 0, 255),
            specular: 500.0,
            reflective: 0.3,
        },
        Sphere {
            center: Vec3 {
                a: -2.0,
                b: 0.0,
                c: 4.0,
            },
            radius: 1.0,
            color: (0, 255, 0),
            specular: 10.0,
            reflective: 0.4,
        },
        Sphere {
            center: Vec3 {
                a: 0.0,
                b: -5001.0,
                c: 0.0,
            },
            radius: 5000.0,
            color: (255, 255, 0),
            specular: 1000.0,
            reflective: 0.5,
        },
    ];
}

fn build_scene_lights() -> Vec<Light> {
    return vec![
        Light {
            light_type: LightType::Ambient,
            intensity: 0.2,
            position: None,
            direction: None,
        },
        Light {
            light_type: LightType::Point,
            intensity: 0.6,
            position: Some(Vec3 {
                a: 2.0,
                b: 1.0,
                c: 0.0,
            }),
            direction: None,
        },
        Light {
            light_type: LightType::Directional,
            intensity: 0.2,
            position: None,
            direction: Some(Vec3 {
                a: 1.0,
                b: 4.0,
                c: 4.0,
            }),
        },
    ];
}

fn send_camera_information_update(
    tx: &std::sync::mpsc::Sender<[Vec3; 3]>,
    camera_pos: Vec3,
    camera_rot: Vec3,
    window_size: Vec3,
) {
    if tx.send([camera_pos, camera_rot, window_size]).is_ok() {}
}

fn start_scene_render_thread(el_proxy: glutin::event_loop::EventLoopProxy<std::vec::Vec<f32>>, rx: std::sync::mpsc::Receiver<[Vec3; 3]>) {
    // Move graphics rendering to separate thread so the main thread can move to the event loop
    // glutin event loop is required to be on main thread for certain platforms (iOS)
    thread::spawn(move || {
        // Render new scene
        let mut window_width = STARTING_SCREEN_WIDTH;
        let mut window_height = STARTING_SCREEN_HEIGHT;
        let mut camera_pos = CAMERA_STARTING_POSITION;
        let mut camera_rot = Vec3 {
            a: CAMERA_STARTING_ROTATION_X_ANGLE,
            b: CAMERA_STARTING_ROTATION_Y_ANGLE,
            c: CAMERA_STARTING_ROTATION_Z_ANGLE,
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
        let vertex_data = render_scene(
            &scene_spheres,
            &scene_lights,
            camera_pos,
            camera_rotation_matrix,
            window_width,
            window_height,
        );
        // Send scene to event loop
        match el_proxy.send_event(vertex_data) {
            Ok(_) => (),
            Err(e) => println!("Error: {}", e),
        }
        if !RENDER_STATIC_IMAGE {
            let mut last_render_time_thread: SystemTime = SystemTime::now();
            loop {
                // Get input from event loop if any
                let r: Option<[Vec3; 3]> = rx.try_iter().last();
                if let Some(received) = r {
                    camera_pos = received[0];
                    camera_rot.a = received[1].a;
                    camera_rot.b = received[1].b;
                    camera_rot.c = received[1].c;
                    window_width = received[2].a as i32;
                    window_height = received[2].b as i32;
                }
                // Render new scene
                let camera_rotation_matrix: RotationMatrix3D =
                    build_rotation_matrix_3d(camera_rot.a, camera_rot.b, camera_rot.c);
                if window_width % 2 != 0 {
                    window_width += 1;
                }
                if window_height % 2 != 0 {
                    window_height += 1;
                }
                let vertex_data = render_scene(
                    &scene_spheres,
                    &scene_lights,
                    camera_pos,
                    camera_rotation_matrix,
                    window_width,
                    window_height,
                );
                match last_render_time_thread.elapsed() {
                    Ok(elapsed) => {
                        if elapsed.as_secs_f32() < MAX_FRAME_TIME_SEC {
                            thread::sleep(time::Duration::from_millis(elapsed.as_millis() as u64));
                        }
                    }
                    Err(e) => {
                        println!("Error: {:?}", e);
                    }
                }
                // Send scene to event loop
                match el_proxy.send_event(vertex_data) {
                    Ok(_) => (last_render_time_thread = SystemTime::now()),
                    Err(e) => println!("Error: {}", e),
                }
            }
        }
    });
}

fn start_event_loop(event_loop: glutin::event_loop::EventLoop<std::vec::Vec<f32>>, tx: std::sync::mpsc::Sender<[Vec3; 3]>) {
    let mut camera_pos = CAMERA_STARTING_POSITION;
    let mut camera_rot = Vec3 {
        a: CAMERA_STARTING_ROTATION_X_ANGLE,
        b: CAMERA_STARTING_ROTATION_Y_ANGLE,
        c: CAMERA_STARTING_ROTATION_Z_ANGLE,
    };
    let mut last_render_time: SystemTime = SystemTime::now();
    let mut vertex_data: Vec<f32> = vec![0.0; 1];

    // Setup opengl window using glutin
    let window: glutin::window::WindowBuilder;
    if START_FULLSCREEN {
        window = glutin::window::WindowBuilder::new()
            .with_title("SpaceTracer")
            .with_inner_size(glutin::dpi::PhysicalSize {
                height: STARTING_SCREEN_HEIGHT * SCREEN_MULTIPLIER,
                width: STARTING_SCREEN_WIDTH * SCREEN_MULTIPLIER,
            })
            .with_fullscreen(Some(glutin::window::Fullscreen::Borderless(None)));
    } else {
        window = glutin::window::WindowBuilder::new()
            .with_title("SpaceTracer")
            .with_inner_size(glutin::dpi::PhysicalSize {
                height: STARTING_SCREEN_HEIGHT * SCREEN_MULTIPLIER,
                width: STARTING_SCREEN_WIDTH * SCREEN_MULTIPLIER,
            })
            .with_maximized(START_MAXIMIZED);
    }
    let gl_window = glutin::ContextBuilder::new()
        .build_windowed(window, &event_loop)
        .unwrap();
    let gl_window = unsafe { gl_window.make_current() }.unwrap();
    gl::load_with(|symbol| gl_window.get_proc_address(symbol));
    let vertex_shader = compile_shader(VS_SRC, gl::VERTEX_SHADER);
    let fragment_shader = compile_shader(FS_SRC, gl::FRAGMENT_SHADER);
    let program = link_program(vertex_shader, fragment_shader);
    let (vao, vbo) = setup_graphics(program);

    event_loop.run(move |event, _, control_flow| {
        use glutin::event::{DeviceEvent, Event, WindowEvent};
        use glutin::event_loop::ControlFlow;
        *control_flow = ControlFlow::Wait;
        match event {
            Event::LoopDestroyed => {}
            Event::WindowEvent { event, .. } => {
                if event == WindowEvent::CloseRequested {
                    unsafe {
                        gl::DeleteProgram(program);
                        gl::DeleteShader(fragment_shader);
                        gl::DeleteShader(vertex_shader);
                        gl::DeleteBuffers(1, &vbo);
                        gl::DeleteVertexArrays(1, &vao);
                    }
                    *control_flow = ControlFlow::Exit
                }
            }
            Event::RedrawRequested(_) => {
                if tx
                    .send([
                        camera_pos,
                        Vec3 {
                            a: camera_rot.a,
                            b: camera_rot.b,
                            c: camera_rot.c,
                        },
                        Vec3 {
                            a: gl_window.window().inner_size().width as f32,
                            b: gl_window.window().inner_size().height as f32,
                            c: 0.0,
                        },
                    ])
                    .is_ok()
                {};
                draw_graphics(&gl_window, &vertex_data);
            }
            Event::UserEvent(vertex_data_new) => {
                draw_graphics(&gl_window, &vertex_data_new);
                vertex_data = vertex_data_new;
            }
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::Added => {}
                DeviceEvent::Removed => {}
                DeviceEvent::MouseMotion { .. } => {}
                DeviceEvent::MouseWheel { .. } => {}
                DeviceEvent::Motion { .. } => {}
                DeviceEvent::Button { .. } => {}
                DeviceEvent::Key(keyboad_input) => match keyboad_input.virtual_keycode {
                    None => {}
                    Some(x) => match x {
                        glutin::event::VirtualKeyCode::D => match last_render_time.elapsed() {
                            Ok(elapsed) => {
                                camera_pos = camera_pos
                                    + Vec3 {
                                        a: CAMERA_MOVEMENT_SPEED
                                            * min(
                                                min(elapsed.as_secs_f32(), MAX_FRAME_TIME_SEC),
                                                MAX_FRAME_TIME_SEC,
                                            ),
                                        b: 0.0,
                                        c: 0.0,
                                    };
                                send_camera_information_update(
                                    &tx,
                                    camera_pos,
                                    camera_rot,
                                    Vec3 {
                                        a: gl_window.window().inner_size().width as f32,
                                        b: gl_window.window().inner_size().height as f32,
                                        c: 0.0,
                                    },
                                );
                                last_render_time = SystemTime::now();
                            }
                            Err(e) => {
                                println!("Error: {:?}", e);
                            }
                        },
                        glutin::event::VirtualKeyCode::A => match last_render_time.elapsed() {
                            Ok(elapsed) => {
                                camera_pos = camera_pos
                                    + Vec3 {
                                        a: -CAMERA_MOVEMENT_SPEED
                                            * min(
                                                min(elapsed.as_secs_f32(), MAX_FRAME_TIME_SEC),
                                                MAX_FRAME_TIME_SEC,
                                            ),
                                        b: 0.0,
                                        c: 0.0,
                                    };
                                send_camera_information_update(
                                    &tx,
                                    camera_pos,
                                    camera_rot,
                                    Vec3 {
                                        a: gl_window.window().inner_size().width as f32,
                                        b: gl_window.window().inner_size().height as f32,
                                        c: 0.0,
                                    },
                                );
                                last_render_time = SystemTime::now();
                            }
                            Err(e) => {
                                println!("Error: {:?}", e);
                            }
                        },
                        glutin::event::VirtualKeyCode::W => match last_render_time.elapsed() {
                            Ok(elapsed) => {
                                camera_pos = camera_pos
                                    + Vec3 {
                                        a: 0.0,
                                        b: CAMERA_MOVEMENT_SPEED
                                            * min(
                                                min(elapsed.as_secs_f32(), MAX_FRAME_TIME_SEC),
                                                MAX_FRAME_TIME_SEC,
                                            ),
                                        c: 0.0,
                                    };
                                send_camera_information_update(
                                    &tx,
                                    camera_pos,
                                    camera_rot,
                                    Vec3 {
                                        a: gl_window.window().inner_size().width as f32,
                                        b: gl_window.window().inner_size().height as f32,
                                        c: 0.0,
                                    },
                                );
                                last_render_time = SystemTime::now();
                            }
                            Err(e) => {
                                println!("Error: {:?}", e);
                            }
                        },
                        glutin::event::VirtualKeyCode::S => match last_render_time.elapsed() {
                            Ok(elapsed) => {
                                camera_pos = camera_pos
                                    + Vec3 {
                                        a: 0.0,
                                        b: -CAMERA_MOVEMENT_SPEED
                                            * min(elapsed.as_secs_f32(), MAX_FRAME_TIME_SEC),
                                        c: 0.0,
                                    };
                                send_camera_information_update(
                                    &tx,
                                    camera_pos,
                                    camera_rot,
                                    Vec3 {
                                        a: gl_window.window().inner_size().width as f32,
                                        b: gl_window.window().inner_size().height as f32,
                                        c: 0.0,
                                    },
                                );
                                last_render_time = SystemTime::now();
                            }
                            Err(e) => {
                                println!("Error: {:?}", e);
                            }
                        },
                        glutin::event::VirtualKeyCode::U => match last_render_time.elapsed() {
                            Ok(elapsed) => {
                                camera_pos = camera_pos
                                    + Vec3 {
                                        a: 0.0,
                                        b: 0.0,
                                        c: CAMERA_MOVEMENT_SPEED
                                            * min(elapsed.as_secs_f32(), MAX_FRAME_TIME_SEC),
                                    };
                                send_camera_information_update(
                                    &tx,
                                    camera_pos,
                                    camera_rot,
                                    Vec3 {
                                        a: gl_window.window().inner_size().width as f32,
                                        b: gl_window.window().inner_size().height as f32,
                                        c: 0.0,
                                    },
                                );
                                last_render_time = SystemTime::now();
                            }
                            Err(e) => {
                                println!("Error: {:?}", e);
                            }
                        },
                        glutin::event::VirtualKeyCode::J => match last_render_time.elapsed() {
                            Ok(elapsed) => {
                                camera_pos = camera_pos
                                    + Vec3 {
                                        a: 0.0,
                                        b: 0.0,
                                        c: -CAMERA_MOVEMENT_SPEED
                                            * min(elapsed.as_secs_f32(), MAX_FRAME_TIME_SEC),
                                    };
                                send_camera_information_update(
                                    &tx,
                                    camera_pos,
                                    camera_rot,
                                    Vec3 {
                                        a: gl_window.window().inner_size().width as f32,
                                        b: gl_window.window().inner_size().height as f32,
                                        c: 0.0,
                                    },
                                );
                                last_render_time = SystemTime::now();
                            }
                            Err(e) => {
                                println!("Error: {:?}", e);
                            }
                        },
                        glutin::event::VirtualKeyCode::Q => {
                            camera_pos = CAMERA_STARTING_POSITION;
                            camera_rot.a = CAMERA_STARTING_ROTATION_X_ANGLE;
                            camera_rot.b = CAMERA_STARTING_ROTATION_Y_ANGLE;
                            camera_rot.c = CAMERA_STARTING_ROTATION_Z_ANGLE;
                            send_camera_information_update(
                                &tx,
                                camera_pos,
                                camera_rot,
                                Vec3 {
                                    a: gl_window.window().inner_size().width as f32,
                                    b: gl_window.window().inner_size().height as f32,
                                    c: 0.0,
                                },
                            );
                        }
                        glutin::event::VirtualKeyCode::P => {
                            camera_rot.a = CAMERA_STARTING_ROTATION_X_ANGLE;
                            camera_rot.b = CAMERA_STARTING_ROTATION_Y_ANGLE;
                            camera_rot.c = CAMERA_STARTING_ROTATION_Z_ANGLE;
                            send_camera_information_update(
                                &tx,
                                camera_pos,
                                camera_rot,
                                Vec3 {
                                    a: gl_window.window().inner_size().width as f32,
                                    b: gl_window.window().inner_size().height as f32,
                                    c: 0.0,
                                },
                            );
                        }
                        glutin::event::VirtualKeyCode::Up => match last_render_time.elapsed() {
                            Ok(elapsed) => {
                                camera_rot.c -= CAMERA_ROTATION_SPEED
                                    * min(elapsed.as_secs_f32(), MAX_FRAME_TIME_SEC);
                                send_camera_information_update(
                                    &tx,
                                    camera_pos,
                                    camera_rot,
                                    Vec3 {
                                        a: gl_window.window().inner_size().width as f32,
                                        b: gl_window.window().inner_size().height as f32,
                                        c: 0.0,
                                    },
                                );
                                last_render_time = SystemTime::now();
                            }
                            Err(e) => {
                                println!("Error: {:?}", e);
                            }
                        },
                        glutin::event::VirtualKeyCode::Down => match last_render_time.elapsed() {
                            Ok(elapsed) => {
                                camera_rot.c += CAMERA_ROTATION_SPEED
                                    * min(elapsed.as_secs_f32(), MAX_FRAME_TIME_SEC);
                                send_camera_information_update(
                                    &tx,
                                    camera_pos,
                                    camera_rot,
                                    Vec3 {
                                        a: gl_window.window().inner_size().width as f32,
                                        b: gl_window.window().inner_size().height as f32,
                                        c: 0.0,
                                    },
                                );
                                last_render_time = SystemTime::now();
                            }
                            Err(e) => {
                                println!("Error: {:?}", e);
                            }
                        },
                        glutin::event::VirtualKeyCode::Right => match last_render_time.elapsed() {
                            Ok(elapsed) => {
                                camera_rot.b += CAMERA_ROTATION_SPEED
                                    * min(elapsed.as_secs_f32(), MAX_FRAME_TIME_SEC);
                                send_camera_information_update(
                                    &tx,
                                    camera_pos,
                                    camera_rot,
                                    Vec3 {
                                        a: gl_window.window().inner_size().width as f32,
                                        b: gl_window.window().inner_size().height as f32,
                                        c: 0.0,
                                    },
                                );
                                last_render_time = SystemTime::now();
                            }
                            Err(e) => {
                                println!("Error: {:?}", e);
                            }
                        },
                        glutin::event::VirtualKeyCode::Left => match last_render_time.elapsed() {
                            Ok(elapsed) => {
                                camera_rot.b -= CAMERA_ROTATION_SPEED
                                    * min(elapsed.as_secs_f32(), MAX_FRAME_TIME_SEC);
                                send_camera_information_update(
                                    &tx,
                                    camera_pos,
                                    camera_rot,
                                    Vec3 {
                                        a: gl_window.window().inner_size().width as f32,
                                        b: gl_window.window().inner_size().height as f32,
                                        c: 0.0,
                                    },
                                );
                                last_render_time = SystemTime::now();
                            }
                            Err(e) => {
                                println!("Error: {:?}", e);
                            }
                        },
                        glutin::event::VirtualKeyCode::I => match last_render_time.elapsed() {
                            Ok(elapsed) => {
                                camera_rot.a += CAMERA_ROTATION_SPEED
                                    * min(elapsed.as_secs_f32(), MAX_FRAME_TIME_SEC);
                                send_camera_information_update(
                                    &tx,
                                    camera_pos,
                                    camera_rot,
                                    Vec3 {
                                        a: gl_window.window().inner_size().width as f32,
                                        b: gl_window.window().inner_size().height as f32,
                                        c: 0.0,
                                    },
                                );
                                last_render_time = SystemTime::now();
                            }
                            Err(e) => {
                                println!("Error: {:?}", e);
                            }
                        },
                        glutin::event::VirtualKeyCode::K => match last_render_time.elapsed() {
                            Ok(elapsed) => {
                                camera_rot.a -= CAMERA_ROTATION_SPEED
                                    * min(elapsed.as_secs_f32(), MAX_FRAME_TIME_SEC);
                                send_camera_information_update(
                                    &tx,
                                    camera_pos,
                                    camera_rot,
                                    Vec3 {
                                        a: gl_window.window().inner_size().width as f32,
                                        b: gl_window.window().inner_size().height as f32,
                                        c: 0.0,
                                    },
                                );
                                last_render_time = SystemTime::now();
                            }
                            Err(e) => {
                                println!("Error: {:?}", e);
                            }
                        },
                        glutin::event::VirtualKeyCode::F => {
                            gl_window
                                .window()
                                .set_fullscreen(Some(glutin::window::Fullscreen::Borderless(None)));
                        }
                        glutin::event::VirtualKeyCode::Escape => {
                            gl_window.window().set_fullscreen(None);
                        }
                        _ => {}
                    },
                },
                DeviceEvent::Text { .. } => {}
            },
            _ => (),
        }
    });
}

fn main() {
    // Setup communication between threads
    let (tx, rx) = mpsc::channel();

    // Setup event loop
    let event_loop = glutin::event_loop::EventLoop::<Vec<f32>>::with_user_event();
    let el_proxy = event_loop.create_proxy();

    start_scene_render_thread(el_proxy, rx);
    start_event_loop(event_loop, tx);
}
