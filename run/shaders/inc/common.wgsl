struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

struct MainUniform {
    // at frame no 0, texture 1 should be used and texture 0 should be drawn on
    frame_no: u32,
    current_instant: f32,
    seed_0: u32,
    seed_1: u32,
    seed_2: u32,
    seed_3: u32,
    visualisation_mode: i32,
}

struct GeometryElement {
    normal_and_depth: vec4<f32>,
    albedo: vec4<f32>,
    position: vec4<f32>,
}

fn gb_idx_i(coords: vec2<i32>) -> i32 {
    let cols = textureDimensions(texture_rt).x;
    return coords.x + coords.y * i32(cols);
}

fn gb_idx_u(coords: vec2<u32>) -> u32 {
    let cols = textureDimensions(texture_rt).x;
    return coords.x + coords.y * cols;
}
