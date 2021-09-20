use std::sync::mpsc;

use space_tracer::{start_scene_render_thread, start_event_loop};

fn main() {
    // Setup communication between threads
    let (tx, rx) = mpsc::channel();

    // Setup event loop
    let event_loop = glutin::event_loop::EventLoop::<Vec<f32>>::with_user_event();
    let el_proxy = event_loop.create_proxy();

    start_scene_render_thread(el_proxy, rx);
    start_event_loop(event_loop, tx);
}
