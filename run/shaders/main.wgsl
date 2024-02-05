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

@group(1) @binding(0) var rt_sampler: sampler;
@group(1) @binding(1) var texture_rt: texture_2d<f32>;
@group(1) @binding(2) var texture_normal: texture_2d<f32>;
@group(1) @binding(3) var texture_pos: texture_2d<f32>;
@group(1) @binding(4) var texture_denoise_0: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(5) var texture_denoise_1: texture_storage_2d<rgba8unorm, read_write>;

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

fn a_trous(tex_coords: vec2<i32>, tex_dims: vec2<i32>, step_scale: i32) -> vec3<f32> {
    /* abc
       bbc
       ccc */
    // js for testing stuff:
    // let g=f=>{let a=[];for(let y=-2;y<=2;y++){let b=[];for(let x=-2;x<=2;x++){b.push(f(x, y))}a.push(b)}return a}
    // let min = (x,y)=> x < y ? x : y;
    // let max = (x,y)=> x < y ? y : x;
    // let clamp = (v,lo,hi) => max(min(v, hi), lo);
    // let abs = v => v < 0 ? -v : v;
    // g((x,y)=>['a','b','c'][2 - clamp(2 - abs(x), 0, 2 - abs(y))])
    var kernel = array<f32, 3>(1./16., 1./4., 3./8.);

    let center_rt = textureLoad(texture_rt, tex_coords, 0).xyz;
    let center_normal = textureLoad(texture_normal, tex_coords, 0).xyz;
    let center_pos = textureLoad(texture_pos, tex_coords, 0).xyz;

    var sum = vec3<f32>(0.);
    var kernel_sum = 0.;

    let σ_rt = 0.85;
    let σ_n  = 0.3;
    let σ_p  = 0.25;

    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            //let kernel_weight = kernel[(x + 2) + ((y + 2) * 5)];
            let kernel_weight = kernel[2 - clamp(2 - abs(x), 0, 2 - abs(y))];

            let offset = vec2<i32>(x, y) * step_scale;
            let cur_coords = clamp(tex_coords + offset, vec2<i32>(0), tex_dims);

            let sample_rt = textureLoad(texture_rt, cur_coords, 0).xyz;
            let dist_rt = distance(center_rt, sample_rt);
            let weight_rt = min(exp(-dist_rt / (σ_rt * σ_rt)), 1.);

            let sample_normal = textureLoad(texture_normal, cur_coords, 0).xyz;
            let dist_normal = distance(center_normal, sample_normal);
            let weight_normal = min(exp(-dist_normal / (σ_n * σ_n)), 1.);

            let sample_pos = textureLoad(texture_pos, cur_coords, 0).xyz;
            let dist_pos = distance(center_pos, sample_pos);
            let weight_pos = min(exp(-dist_pos / (σ_p * σ_p)), 1.);

            let weight = kernel_weight * weight_rt * weight_normal * weight_pos;

            sum += weight * sample_rt.xyz;
            kernel_sum += weight;
        }
    }

    return sum / kernel_sum;
}

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = textureDimensions(texture_rt);
    let tex_pos = vec2<u32>(vec2<f32>(tex_size) * in.tex_coords);

    //return textureSample(texture_rt, rt_sampler, in.tex_coords);
    //return textureLoad(texture_rt, tex_pos, 0);
    return vec4<f32>(a_trous(vec2<i32>(tex_pos), vec2<i32>(tex_size), 1), 1.);
}
