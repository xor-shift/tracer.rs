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
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = VISUALISER_UVS[VISUALISER_INDICES[vertex_index]];
    out.position = vec4<f32>(VISUALISER_VERTICES[VISUALISER_INDICES[vertex_index]], 0., 1.0);
    return out;
}

struct MainUniform {
    width: u32,                 // 00..03
    height: u32,                // 04..07
    visualisation_mode: i32,    // 08..0B
}

@group(0) @binding(0) var<uniform> uniforms: MainUniform;
@group(1) @binding(0) var texture_rt: texture_2d<f32>;
@group(1) @binding(1) var texture_geo_pack_0: texture_2d<u32>;
@group(1) @binding(2) var texture_geo_pack_1: texture_2d<u32>;
@group(1) @binding(3) var texture_geo_pack_0_old: texture_2d<u32>;
@group(1) @binding(4) var texture_geo_pack_1_old: texture_2d<u32>;
@group(1) @binding(5) var texture_denoise_0: texture_2d<f32>;
@group(1) @binding(6) var texture_denoise_1: texture_2d<f32>;

fn collect_geo_u(coords: vec2<u32>) -> GeometryElement { return collect_geo_t2d(coords, texture_geo_pack_0, texture_geo_pack_1); }

fn aces_film(x: vec3<f32>) -> vec3<f32> {
    let raw = (x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14);
    return clamp(raw, vec3<f32>(0.), vec3<f32>(1.));
}

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = textureDimensions(texture_rt);
    let tex_pos = vec2<u32>(vec2<f32>(tex_size) * in.tex_coords);

    //return vec4<f32>(vec2<f32>(tex_pos) / vec2<f32>(tex_size), 0., 1.);
    //return vec4<f32>(vec2<f32>(tex_pos) / vec2<f32>(f32(uniforms.width), f32(uniforms.height)), 0., 1.);
    //return vec4<f32>(ge_normal(geometry_buffer[gb_idx_u(tex_pos)]), 1.);
    //return vec4<f32>(geometry_buffer[gb_idx_u(tex_pos)].albedo_and_origin_dist.xyz, 1.);

    //return vec4<f32>(TINDEX_COLORS[geometry_buffer[tex_pos.x + tex_pos.y * uniforms.width].triangle_index], 1.);

    let geometry = collect_geo_u(tex_pos);
    let old_geometry = collect_geo_t2d(tex_pos, texture_geo_pack_0_old, texture_geo_pack_1_old);
    
    switch uniforms.visualisation_mode {
        case 0 : { return vec4<f32>(aces_film(textureLoad(texture_rt, tex_pos, 0).xyz), 1.); } // rt
        case 1 : { return vec4<f32>(vec3<f32>(geometry.variance / 1.), 1.); }                  // variance
        case 2 : { return vec4<f32>(aces_film(textureLoad(texture_denoise_0, tex_pos, 0).xyz), 1.); } // denoise 0
        case 3 : { return vec4<f32>(aces_film(textureLoad(texture_denoise_1, tex_pos, 0).xyz), 1.); } // denoise 1
        case 4 : { return vec4<f32>(aces_film(textureLoad(texture_rt, tex_pos, 0).xyz * geometry.albedo), 1.); }        // rt * albedo
        case 5 : { return vec4<f32>(aces_film(textureLoad(texture_denoise_0, tex_pos, 0).xyz * geometry.albedo), 1.); } // denoise 0 * albedo
        case 6 : { return vec4<f32>(aces_film(textureLoad(texture_denoise_1, tex_pos, 0).xyz * geometry.albedo), 1.); } // denoise 1 * albedo
        case 7 : { return vec4<f32>(geometry.albedo, 1.); }                                // albedo
        case 8 : { return vec4<f32>(geometry.normal / 1.5, 1.); }                          // normal
        case 9 : { return vec4<f32>(abs(geometry.normal) / 1.5, 1.); }                     // abs normal
        case 10: { return vec4<f32>(vec3<f32>(geometry.depth), 1.); }                      // depth
        case 11: { return vec4<f32>(geometry.position / 2., 1.); }                         // scene location
        case 12: { return vec4<f32>(abs(geometry.position) / 2., 1.); }                    // abs scene location
        case 13: { return vec4<f32>(vec3<f32>(geometry.distance_from_origin / 50.), 1.); } // dist from origin
        case 14: { return vec4<f32>(get_tindex_color(geometry.object_index), 1.); }        // object index
        default: { return vec4<f32>(0., 0., 0., 1.); }
    }
}
