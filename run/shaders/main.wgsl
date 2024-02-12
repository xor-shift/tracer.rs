struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

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

@vertex fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    // vert: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = VISUALISER_UVS[VISUALISER_INDICES[vertex_index]];
    out.position = vec4<f32>(VISUALISER_VERTICES[VISUALISER_INDICES[vertex_index]], 0., 1.0);
    return out;
}

@group(1) @binding(0) var texture_rt: texture_2d<f32>;
@group(1) @binding(1) var<storage, read> geometry_buffer: array<GeometryElement>;
@group(1) @binding(2) var texture_denoise_0: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(3) var texture_denoise_1: texture_storage_2d<rgba8unorm, read_write>;

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = textureDimensions(texture_rt);
    let tex_pos = vec2<u32>(vec2<f32>(tex_size) * in.tex_coords);

    switch uniforms.visualisation_mode {
        // indirect light
        case 0 : { return textureLoad(texture_rt, tex_pos, 0); }
        // direct light
        case 1 : { return vec4<f32>(geometry_buffer[gb_idx_u(tex_pos)].direct_illum, 1.); }
        // indirect light composited with albedo
        case 2 : { return vec4<f32>(textureLoad(texture_rt, tex_pos, 0).xyz * ge_albedo(geometry_buffer[gb_idx_u(tex_pos)]), 1.); }

        case 3 : { return vec4<f32>(textureLoad(texture_denoise_0, tex_pos).xyz, 1.); }
        case 4 : { return vec4<f32>(textureLoad(texture_denoise_0, tex_pos).xyz, 1.); }

        // filtered indirect light composited with albedo
        case 5 : { return vec4<f32>(textureLoad(texture_denoise_0, tex_pos).xyz * ge_albedo(geometry_buffer[gb_idx_u(tex_pos)]), 1.); }
        case 6 : { return vec4<f32>(textureLoad(texture_denoise_1, tex_pos).xyz * ge_albedo(geometry_buffer[gb_idx_u(tex_pos)]), 1.); }

        // albedo
        case 7 : { return vec4<f32>(ge_albedo(geometry_buffer[gb_idx_u(tex_pos)]), 1.); }

        // normals, absolute normals
        case 8 : { return vec4<f32>(ge_normal(geometry_buffer[gb_idx_u(tex_pos)]), 1.); }
        case 9 : { return vec4<f32>(abs(ge_normal(geometry_buffer[gb_idx_u(tex_pos)])), 1.); }

        //position, distance
        //case 7 : { return vec4<f32>(ge_position(geometry_buffer[gb_idx_u(tex_pos)]) / 10., 1.); }
        case 10: { return vec4<f32>(vec3<f32>(ge_origin_distance(geometry_buffer[gb_idx_u(tex_pos)]) / 100.), 1.); }

        //depth
        case 11: { return vec4<f32>(vec3<f32>((ge_depth(geometry_buffer[gb_idx_u(tex_pos)]) - 10.) / 10.), 1.); }

        default: { return vec4<f32>(0., 0., 0., 1.); }
    }
}
