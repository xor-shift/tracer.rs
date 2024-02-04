struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex fn vs_main(vert: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = vert.tex_coords;
    out.position = vec4<f32>(vert.position, 1.0);
    return out;
}

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

fn fs_do_sample(uv: vec2<f32>, texture: texture_storage_2d<rgba8unorm, read_write>) -> vec4<f32> {
    let tex_size = textureDimensions(texture_0);
    let tex_pos = vec2<u32>(vec2<f32>(tex_size) * uv);

    let sample = textureLoad(texture_0, tex_pos);

    return sample;
}

// Converts a color from linear light gamma to sRGB gamma
fn linear_to_srgb(linear: vec4<f32>) -> vec4<f32>{
    let cutoff = linear.rgb < vec3(0.0031308);
    let higher = vec3(1.055) * pow(linear.rgb, vec3(1.0/2.4)) - vec3(0.055);
    let lower = linear.rgb * vec3(12.92);

    return vec4(mix(higher, lower, vec3<f32>(cutoff)), linear.a);
}

// Converts a color from sRGB gamma to linear light gamma
fn srgb_to_linear(srgb: vec4<f32>) -> vec4<f32> {
    let cutoff = srgb.rgb < vec3(0.04045);
    let higher = pow((srgb.rgb + vec3(0.055))/vec3(1.055), vec3(2.4));
    let lower = srgb.rgb/vec3(12.92);

    return vec4(mix(higher, lower, vec3<f32>(cutoff)), srgb.a);
}

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    //let sample_0 = textureSample(texture_0, tex_sampler, in.tex_coords);
    //let sample_1 = textureSample(texture_1, tex_sampler, in.tex_coords);

    let sample_0 = fs_do_sample(in.tex_coords, texture_0);
    let sample_1 = fs_do_sample(in.tex_coords, texture_1);

    let texture_selection = select(0u, 1u, uniforms.frame_no % 2u == 1u);
    let sample = select(sample_0, sample_1, texture_selection == 0u);

    return sample;
}
