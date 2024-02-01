struct MainUniform {
    // at frame no 0, texture 1 should be used and texture 0 should be drawn on
    frame_no: u32,
    current_instant: f32,
    seed_0: u32,
    seed_1: u32,
    seed_2: u32,
    seed_3: u32,
};

@group(0) @binding(0) var<uniform> uniforms: MainUniform;
@group(1) @binding(0) var texture_0: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(1) var texture_1: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(2) var texture_noise: texture_storage_2d<rgba32uint, read>;

fn actual_cs(pixel: vec2<u32>, dimensions: vec2<u32>, rng: ptr<function, RNGState>) -> vec4<f32> {
    let camera = PinpointCamera(FRAC_PI_4);
    let ray = pinpoint_generate_ray(camera, pixel, dimensions, vec3<f32>(0.), rng);

    let rng_res = vec3<f32>(u32_to_f32_unorm(rng_next(rng)));
    return vec4<f32>(rng_res, 1.0);

    /*let max_iterations = 50u;
    var final_iteration = max_iterations;
    
    var c = (vec2<f32>(pixel) / vec2<f32>(dimensions)) * 3.0 - vec2<f32>(2.25, 1.5);
    var current_z = c;
    
    for (var i = 0u; i < max_iterations; i++) {
        let next = vec2<f32>(
            (current_z.x * current_z.x - current_z.y * current_z.y) + c.x,
            (2.0 * current_z.x * current_z.y) + c.y
        );

        current_z = next;

        if length(current_z) > 4.0 {
            final_iteration = i;
            break;
        }
    }
    let value = f32(final_iteration) / f32(max_iterations);

    return vec4<f32>(value, value, value, 1.);*/

    //let texture_uv = vec2<f32>(pixel) / vec2<f32>(dimensions);
    //return vec4<f32>(texture_uv, (sin(uniforms.current_instant) + 1.) / 2., 1.);
}
 
@compute @workgroup_size(1) fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    let texture_selection = select(0u, 1u, uniforms.frame_no % 2u == 0u);
    let texture_dimensions = select(textureDimensions(texture_0), textureDimensions(texture_1), texture_selection == 0u);

    /*let uv = vec2<f32>(pixel) / vec2<f32>(texture_dimensions);
    let pix_hash = vec2_u32_hash(pixel);
    let rng_pix_seed = sample_noise(uv);
    var rng = RNGState(array<u32, 4>(
        uniforms.seed_0 ^ pix_hash ^ rng_pix_seed.r,
        uniforms.seed_1 ^ pix_hash ^ rng_pix_seed.g,
        uniforms.seed_2 ^ pix_hash ^ rng_pix_seed.b,
        uniforms.seed_3 ^ pix_hash ^ rng_pix_seed.a
    ));*/

    var rng = setup_rng(pixel, texture_dimensions);

    /*rng.state[0] = rng_next(&rng);
    rng.state[1] = rng_next(&rng);
    rng.state[2] = rng_next(&rng);
    rng.state[3] = rng_next(&rng);*/

    let out_color = actual_cs(global_id.xy, texture_dimensions, &rng);

    if texture_selection == 0u {
        textureStore(texture_0, vec2<i32>(global_id.xy), out_color);
    } else {
        textureStore(texture_1, vec2<i32>(global_id.xy), out_color);
    }
}
