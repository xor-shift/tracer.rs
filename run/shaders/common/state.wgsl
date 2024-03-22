struct State {
    camera_transform: mat4x4<f32>,
    inverse_transform: mat4x4<f32>,
    frame_seed: vec4<u32>,
    camera_position: vec3<f32>,
    frame_no: u32,
    dimensions: vec2<u32>,
    current_instant: f32,
    visualisation_mode: i32,
}
