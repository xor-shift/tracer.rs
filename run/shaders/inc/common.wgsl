struct MainUniform {
    width: u32,                 // 00..03
    height: u32,                // 04..07
    frame_no: u32,              // 08..0B
    current_instant: f32,       // 0C..0F
    seed_0: u32,                // 10..13
    seed_1: u32,                // 14..17
    seed_2: u32,                // 18..1B
    seed_3: u32,                // 1C..1F
    visualisation_mode: i32,    // 20..23
    camera_position: vec3<f32>, // 30..3B
}

@group(0) @binding(0) var<uniform> uniforms: MainUniform;

struct GeometryElement {
    normal_and_depth: vec4<f32>,
    albedo_and_origin_dist: vec4<f32>,
    direct_illum: vec3<f32>,
}

fn ge_normal(ge: GeometryElement) -> vec3<f32> { return ge.normal_and_depth.xyz; }
fn ge_depth(ge: GeometryElement) -> f32 { return ge.normal_and_depth.w; }
fn ge_albedo(ge: GeometryElement) -> vec3<f32> { return ge.albedo_and_origin_dist.xyz; }
fn ge_origin_distance(ge: GeometryElement) -> f32 { return ge.albedo_and_origin_dist.w; }
//fn ge_position(ge: GeometryElement) -> vec3<f32> { return ge.position.xyz; }

fn gb_idx_i(coords: vec2<i32>) -> i32 {
    // let cols = textureDimensions(texture_rt).x;
    return coords.x + coords.y * i32(uniforms.width);
}

fn gb_idx_u(coords: vec2<u32>) -> u32 {
    // let cols = textureDimensions(texture_rt).x;
    return coords.x + coords.y * uniforms.width;
}

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
