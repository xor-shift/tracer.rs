@vertex fn vs_main(vert: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = vert.tex_coords;
    out.position = vec4<f32>(vert.position, 1.0);
    return out;
}

@group(0) @binding(0) var<uniform> uniforms: MainUniform;

@group(1) @binding(0) var texture_rt: texture_2d<f32>;
@group(1) @binding(1) var<storage, read> geometry_buffer: array<GeometryElement>;
@group(1) @binding(2) var texture_denoise_0: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(3) var texture_denoise_1: texture_storage_2d<rgba8unorm, read_write>;

fn linear_to_srgb(linear: vec4<f32>) -> vec4<f32>{
    let cutoff = linear.rgb < vec3(0.0031308);
    let higher = vec3(1.055) * pow(linear.rgb, vec3(1.0/2.4)) - vec3(0.055);
    let lower = linear.rgb * vec3(12.92);

    return vec4(mix(higher, lower, vec3<f32>(cutoff)), linear.a);
}

fn srgb_to_linear(srgb: vec4<f32>) -> vec4<f32> {
    let cutoff = srgb.rgb < vec3(0.04045);
    let higher = pow((srgb.rgb + vec3(0.055))/vec3(1.055), vec3(2.4));
    let lower = srgb.rgb/vec3(12.92);

    return vec4(mix(higher, lower, vec3<f32>(cutoff)), srgb.a);
}

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = textureDimensions(texture_rt);
    let tex_pos = vec2<u32>(vec2<f32>(tex_size) * in.tex_coords);

    switch uniforms.visualisation_mode {
        // indirect light
        case 0 : { return textureLoad(texture_rt, tex_pos, 0); }
        // indirect light composited with albedo
        case 1 : { return vec4<f32>(textureLoad(texture_rt, tex_pos, 0).xyz * geometry_buffer[gb_idx_u(tex_pos)].albedo.xyz, 1.); }

        // filtered indirect light composited with albedo
        case 2 : { return vec4<f32>(textureLoad(texture_denoise_0, tex_pos).xyz * geometry_buffer[gb_idx_u(tex_pos)].albedo.xyz, 1.); }
        case 3 : { return vec4<f32>(textureLoad(texture_denoise_1, tex_pos).xyz * geometry_buffer[gb_idx_u(tex_pos)].albedo.xyz, 1.); }

        // albedo
        case 4 : { return vec4<f32>(geometry_buffer[gb_idx_u(tex_pos)].albedo.xyz, 1.); }

        // normals, absolute normals
        case 5 : { return vec4<f32>(geometry_buffer[gb_idx_u(tex_pos)].normal_and_depth.xyz, 1.); }
        case 6 : { return vec4<f32>(abs(geometry_buffer[gb_idx_u(tex_pos)].normal_and_depth.xyz), 1.); }

        //position
        case 7 : { return vec4<f32>(geometry_buffer[gb_idx_u(tex_pos)].position.xyz / 10., 1.); }

        //depth
        case 8 : { return vec4<f32>(vec3<f32>((geometry_buffer[gb_idx_u(tex_pos)].normal_and_depth.w - 10.) / 10.), 1.); }

        default: { return vec4<f32>(0., 0., 0., 1.); }
    }
}
