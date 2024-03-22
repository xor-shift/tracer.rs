var<private> VISUALISER_VERTICES: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-1., 1.,),
    vec2<f32>(-1., -1.,),
    vec2<f32>(1., -1.,),
    vec2<f32>(1., 1.,),
);

var<private> VISUALISER_INDICES: array<u32, 6> = array<u32, 6>(0, 1, 2, 2, 3, 0);

var<private> VISUALISER_UVS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0., 0.),
    vec2<f32>(0., 1.),
    vec2<f32>(1., 1.),
    vec2<f32>(1., 0.),
);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@group(0) @binding(0) var<uniform> state: State;
@group(1) @binding(0) var texture_rt: texture_2d<f32>;
@group(1) @binding(1) var texture_geo: texture_2d_array<u32>;
@group(1) @binding(2) var texture_denoise: texture_2d_array<f32>;

@vertex fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = VISUALISER_UVS[VISUALISER_INDICES[vertex_index]];
    out.position = vec4<f32>(VISUALISER_VERTICES[VISUALISER_INDICES[vertex_index]], 0., 1.0);
    return out;
}

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_pos = vec2<u32>(vec2<f32>(state.dimensions) * in.tex_coords);

    let geo_0 = textureLoad(texture_geo, tex_pos, 0, 0);
    let geo_1 = textureLoad(texture_geo, tex_pos, 1, 0);

    let denoise_0 = textureLoad(texture_denoise, tex_pos, 0, 0);
    let denoise_1 = textureLoad(texture_denoise, tex_pos, 1, 0);

    switch state.visualisation_mode {
        //default: { return vec4<f32>(1., 0., 0., 0.); }
        default: { return textureLoad(texture_rt, tex_pos, 0); }
    }
}