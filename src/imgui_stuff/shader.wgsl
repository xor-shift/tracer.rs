struct DrawVert {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: u32,
}

fn get_draw_vert_color(draw_vert: DrawVert) -> vec4<f32> {
    return unpack4x8unorm(draw_vert.color);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct Uniforms {
    dimensions: vec2<u32>,
    padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var main_sampler: sampler;
@group(1) @binding(1) var main_texture: texture_2d<f32>;

@vertex fn vs_main(in: DrawVert) -> VertexOutput {
    let unorm_pos = in.position / vec2<f32>(uniforms.dimensions);
    let flipped_pos = vec2<f32>(unorm_pos.x, 1. - unorm_pos.y);
    let real_pos = flipped_pos * 2. - 1.;

    return VertexOutput(
        /* position */ vec4<f32>(real_pos, 0., 1.),
        /* uv       */ in.uv,
        /* color    */ get_draw_vert_color(in),
    );
}

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color * textureSample(main_texture, main_sampler, in.uv);
}
