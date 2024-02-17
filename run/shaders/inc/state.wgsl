struct State {
    camera_transform: mat4x4<f32>,
    frame_seed: vec4<u32>,
    camera_position: vec3<f32>,
    frame_no: u32,
    current_instant: f32,
    width: u32,
    height: u32,
    visualisation_mode: i32,
}
