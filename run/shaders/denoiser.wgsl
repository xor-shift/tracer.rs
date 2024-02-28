@group(0) @binding(0) var<uniform> stride: i32;
@group(1) @binding(0) var texture_input: texture_2d<f32>;
@group(1) @binding(1) var texture_geo_pack_0: texture_2d<u32>;
@group(1) @binding(2) var texture_geo_pack_1: texture_2d<u32>;
@group(1) @binding(3) var texture_denoise_out: texture_storage_2d<rgba8unorm, read_write>;

fn collect_geo_u(coords: vec2<u32>) -> GeometryElement {
    let sample_pack_0 = textureLoad(texture_geo_pack_0, coords, 0);
    let sample_pack_1 = textureLoad(texture_geo_pack_1, coords, 0);

    return unpack_geo(PackedGeometry(sample_pack_0, sample_pack_1));
}

fn weight_basic_dist(distance: f32, σ: f32) -> f32 {
    return min(exp(-distance / (σ * σ)), 1.);
}

fn weight_basic(p: vec3<f32>, q: vec3<f32>, σ: f32) -> f32 {
    return weight_basic_dist(distance(p, q), σ);
}

fn weight_cosine(p: vec3<f32>, q: vec3<f32>, σ: f32) -> f32 {
    return pow(max(0., dot(p, q)), σ);
}

fn weight_luminance(p: vec3<f32>, q: vec3<f32>, variance_p: f32, σ: f32) -> f32 {
    let ε = 0.0001;
    return exp(-length(p - q) / (σ * sqrt(variance_p) + ε));
}

fn sample_compact_kernel_5x5(kernel: ptr<function, array<f32, 3>>, coords: vec2<i32>) -> f32 {
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

    return (*kernel)[2 - clamp(2 - abs(coords.x), 0, 2 - abs(coords.y))];
}

fn a_trous(tex_coords: vec2<i32>, tex_dims: vec2<i32>, step_scale: i32) -> vec3<f32> {
    var kernel = array<f32, 3>(1./16., 1./4., 3./8.); // small kernel from the original a-trous paper


    let center_rt = textureLoad(texture_input, tex_coords, 0).xyz;
    let center_geo = collect_geo_i(tex_coords);

    var sum = vec3<f32>(0.);
    var kernel_sum = 0.;

    // good values for high spp
    /*let σ_p = 0.4; // position
    let σ_n = 0.5; // normal
    let σ_l = 0.8; // luminance*/
    
    //let σ_p = 1.;   // position
    let σ_p = 0.4;   // position
    let σ_n = 128.; // normal
    let σ_l = 0.8;   // luminance
    //let σ_l = 4.;   // luminance

    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let kernel_weight = sample_compact_kernel_5x5(&kernel, vec2<i32>(x, y));

            let offset = vec2<i32>(x, y) * step_scale;
            let cur_coords = clamp(tex_coords + offset, vec2<i32>(0), tex_dims);
            let cur_geo = collect_geo_i(cur_coords);
            let cur_rt = textureLoad(texture_input, cur_coords, 0).xyz;

            let w_lum = weight_basic(center_rt, cur_rt, σ_l);
            //let w_lum = weight_luminance(center_rt, cur_rt, center_geo.variance, σ_l);
            let w_pos = weight_basic(center_geo.position, cur_geo.position, σ_p);
            //let w_dst = weight_basic_dist(abs(center_geo.distance_from_origin - cur_geo.distance_from_origin), σ_p);
            let w_nrm = weight_cosine(center_geo.normal, cur_geo.normal, σ_n);
            //let w_nrm = weight_basic(center_geo.normal, cur_geo.normal, σ_n);

            let weight = kernel_weight * w_lum * w_nrm * w_pos;

            sum += weight * cur_rt.xyz;
            kernel_sum += weight;
        }
    }

    return sum / kernel_sum;
}

@compute @workgroup_size(8, 8) fn cs_main(
    @builtin(global_invocation_id)   global_id: vec3<u32>,
    @builtin(workgroup_id)           workgroup_id: vec3<u32>,
    @builtin(local_invocation_id)    local_id:  vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let res = a_trous(vec2<i32>(global_id.xy), vec2<i32>(textureDimensions(texture_denoise_out)), stride);
    textureStore(texture_denoise_out, global_id.xy, vec4<f32>(res, 1.));
}