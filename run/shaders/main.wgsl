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
@group(1) @binding(1) var texture_albedo: texture_2d<f32>;
@group(1) @binding(2) var texture_pack_normal_depth: texture_2d<f32>;
@group(1) @binding(3) var texture_pack_pos_dist: texture_2d<f32>;
//@group(1) @binding(1) var<storage, read> geometry_buffer: array<GeometryElement>;
@group(1) @binding(4) var texture_denoise_0: texture_2d<f32>;
@group(1) @binding(5) var texture_denoise_1: texture_2d<f32>;

var<private> TINDEX_COLORS: array<vec3<f32>, 7> = array<vec3<f32>, 7>(
    vec3<f32>(1., 0., 0.),
    vec3<f32>(0., 1., 0.),
    vec3<f32>(1., 1., 0.),
    vec3<f32>(0., 0., 1.),
    vec3<f32>(1., 0., 1.),
    vec3<f32>(0., 1., 1.),
    vec3<f32>(1., 1., 1.),
);

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = textureDimensions(texture_rt);
    let tex_pos = vec2<u32>(vec2<f32>(tex_size) * in.tex_coords);

    //return vec4<f32>(vec2<f32>(tex_pos) / vec2<f32>(tex_size), 0., 1.);
    //return vec4<f32>(vec2<f32>(tex_pos) / vec2<f32>(f32(uniforms.width), f32(uniforms.height)), 0., 1.);
    //return vec4<f32>(ge_normal(geometry_buffer[gb_idx_u(tex_pos)]), 1.);
    //return vec4<f32>(geometry_buffer[gb_idx_u(tex_pos)].albedo_and_origin_dist.xyz, 1.);

    //return vec4<f32>(TINDEX_COLORS[geometry_buffer[tex_pos.x + tex_pos.y * uniforms.width].triangle_index], 1.);

    /*switch uniforms.visualisation_mode {
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
    }*/
    
    switch uniforms.visualisation_mode {
        case 0: { return vec4<f32>(textureLoad(texture_rt, tex_pos, 0).xyz, 1.); }                           // rt
        case 1: { return vec4<f32>(textureLoad(texture_denoise_0, tex_pos, 0).xyz, 1.); }                    // denoise 0
        case 2: { return vec4<f32>(textureLoad(texture_denoise_1, tex_pos, 0).xyz, 1.); }                    // denoise 1
        case 3: { return vec4<f32>(textureLoad(texture_albedo, tex_pos, 0).xyz, 1.); }                       // albedo
        case 4: { return vec4<f32>(textureLoad(texture_pack_normal_depth, tex_pos, 0).xyz, 1.); }            // normal
        case 5: { return vec4<f32>(abs(textureLoad(texture_pack_normal_depth, tex_pos, 0).xyz), 1.); }       // abs normal
        case 6: { return vec4<f32>(vec3<f32>(textureLoad(texture_pack_normal_depth, tex_pos, 0).w), 1.); }   // depth
        case 7: { return vec4<f32>(textureLoad(texture_pack_pos_dist, tex_pos, 0).xyz / 50., 1.); }          // scene location
        case 8: { return vec4<f32>(abs(textureLoad(texture_pack_pos_dist, tex_pos, 0).xyz) / 50., 1.); }     // abs scene location
        case 9: { return vec4<f32>(vec3<f32>(textureLoad(texture_pack_pos_dist, tex_pos, 0).w / 50.), 1.); } // dist from origin
        default: { return vec4<f32>(0., 0., 0., 1.); }
    }
}
