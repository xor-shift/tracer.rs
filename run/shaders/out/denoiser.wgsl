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

var<private> TINDEX_COLORS: array<vec3<f32>, 7> = array<vec3<f32>, 7>(
    vec3<f32>(1., 0., 0.),
    vec3<f32>(0., 1., 0.),
    vec3<f32>(1., 1., 0.),
    vec3<f32>(0., 0., 1.),
    vec3<f32>(1., 0., 1.),
    vec3<f32>(0., 1., 1.),
    vec3<f32>(1., 1., 1.),
);

fn get_tindex_color(index: u32) -> vec3<f32> {
    return TINDEX_COLORS[index % 7u];
}
/*struct GeometryElement {
    normal_and_depth: vec4<f32>,
    albedo_and_origin_dist: vec4<f32>,
    scene_position: vec3<f32>,
    triangle_index: u32,
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
}*/

struct GeometryElement {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    depth: f32,
    position: vec3<f32>,
    distance_from_origin: f32,
    object_index: u32,
}

fn collect_geo_i(coords: vec2<i32>) -> GeometryElement {
    return collect_geo_u(vec2<u32>(max(coords, vec2<i32>(0))));
}
const PI: f32 = 3.14159265358979323846264338327950288; // π
const FRAC_PI_2: f32 = 1.57079632679489661923132169163975144; // π/2
const FRAC_PI_3: f32 = 1.04719755119659774615421446109316763; // π/3
const FRAC_PI_4: f32 = 0.785398163397448309615660845819875721; // π/4
const FRAC_PI_6: f32 = 0.39269908169872415480783042290993786; // π/6
const FRAC_1_PI: f32 = 0.318309886183790671537767526745028724; // 1/π
const FRAC_1_SQRT_PI: f32 = 0.564189583547756286948079451560772586; // 1/√π
const FRAC_2_PI: f32 = 0.636619772367581343075535053490057448; // 2/π
const FRAC_2_SQRT_PI: f32 = 1.12837916709551257389615890312154517; // 2/√π
const PHI: f32 = 1.618033988749894848204586834365638118; // φ
const SQRT_2: f32 = 1.41421356237309504880168872420969808; // √2
const FRAC_1_SQRT_2: f32 = 0.707106781186547524400844362104849039; // 1/√2
const SQRT_3: f32 = 1.732050807568877293527446341505872367; // √3
const FRAC_1_SQRT_3: f32 = 0.577350269189625764509148780501957456; // 1/√3

const MAT3x3_IDENTITY: mat3x3<f32> = mat3x3<f32>(1., 0., 0., 0., 1., 0., 0., 0., 1.);

const INF: f32 = 999999999999999999999.;
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