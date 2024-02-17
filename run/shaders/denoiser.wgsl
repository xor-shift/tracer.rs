@group(0) @binding(0) var<uniform> stride: i32;
@group(1) @binding(0) var texture_input: texture_2d<f32>;
@group(1) @binding(1) var geo_texture_albedo: texture_2d<f32>;
@group(1) @binding(2) var geo_texture_pack_normal_depth: texture_2d<f32>;
@group(1) @binding(3) var geo_texture_pack_pos_dist: texture_2d<f32>;
@group(1) @binding(4) var geo_texture_object_index: texture_2d<u32>;
@group(1) @binding(5) var texture_denoise_out: texture_storage_2d<rgba8unorm, read_write>;

fn collect_geo_u(coords: vec2<u32>) -> GeometryElement {
    let sample_albedo = textureLoad(geo_texture_albedo, coords, 0);
    let sample_normal_depth = textureLoad(geo_texture_pack_normal_depth, coords, 0);
    let sample_pos_dist = textureLoad(geo_texture_pack_pos_dist, coords, 0);
    let sample_object_index = textureLoad(geo_texture_object_index, coords, 0);

    return GeometryElement (
        sample_albedo.xyz,
        sample_normal_depth.xyz,
        sample_normal_depth.w,
        sample_pos_dist.xyz,
        sample_pos_dist.w,
        sample_object_index.r,
    );
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

    let center_rt = textureLoad(texture_input, tex_coords, 0).xyz;
    let center_geo = collect_geo_i(tex_coords);

    var sum = vec3<f32>(0.);
    var kernel_sum = 0.;

    let σ_p = 0.7;   // position
    let σ_n = 128.; // normal
    let σ_l = 0.5;   // luminance

    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            //let kernel_weight = kernel[(x + 2) + ((y + 2) * 5)];
            let kernel_weight = kernel[2 - clamp(2 - abs(x), 0, 2 - abs(y))];

            let offset = vec2<i32>(x, y) * step_scale;
            let cur_coords = clamp(tex_coords + offset, vec2<i32>(0), tex_dims);
            let cur_sample = collect_geo_i(cur_coords);

            let sample_rt = textureLoad(texture_input, cur_coords, 0).xyz;
            let dist_rt = distance(center_rt, sample_rt);
            let weight_rt = min(exp(-dist_rt / (σ_l * σ_l)), 1.);

            /*let sample_normal = cur_sample.normal;
            let dist_normal = distance(center_geo.normal, sample_normal);
            let weight_normal = min(exp(-dist_normal / (σ_n * σ_n)), 1.);*/

            let weight_normal = pow(max(0., dot(cur_sample.normal, center_geo.normal)), σ_n);

            let sample_pos = cur_sample.position;
            let dist_pos = distance(center_geo.position, sample_pos);
            let weight_pos = min(exp(-dist_pos / (σ_p * σ_p)), 1.);

            /*let sample_distance = cur_sample.distance_from_origin;
            let dist_distance = abs(sample_distance - center_geo.distance_from_origin);
            let weight_distance = min(exp(-dist_distance / (σ_p * σ_p)), 1.);*/

            let weight = kernel_weight * weight_rt * weight_normal * weight_pos;

            sum += weight * sample_rt.xyz;
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