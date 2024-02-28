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
struct GeometryElement {
    albedo: vec3<f32>,
    variance: f32,
    normal: vec3<f32>,
    depth: f32,
    position: vec3<f32>,
    distance_from_origin: f32,
    object_index: u32,
    was_invalidated: bool,
    similarity_score: f32,
}

/*
Layout:
[albedo r][albedo g][albedo b][]
[[normal θ]][[normal φ]]
[[variance]][[depth]]
[[[[position X]]]]

[[[[position Y]]]]
[[[[position Z]]]]
[bitflags of no specific purpose][[[object index]]]
[[[[]]]]
*/
struct PackedGeometry {
    pack_0: vec4<u32>,
    pack_1: vec4<u32>,
}

fn normal_to_spherical(normal: vec3<f32>) -> vec2<f32> {
    let x = normal.x;
    let y = normal.y;
    let z = normal.z;

    let r = 1.; // sqrt(x*x + y*y + z*z)

    let θ = select(
        acos(z / r),
        FRAC_PI_2,
        x == 0. && z == 0.
    );

    let φ = select(
        sign(y) * acos(x / sqrt(x * x + y * y)),
        -FRAC_PI_2,
        x == 0. && y == 0.
    );

    return vec2<f32>(θ, φ);
}

fn normal_from_spherical(spherical: vec2<f32>) -> vec3<f32> {
    let r = 1.;
    let θ = spherical.x;
    let φ = spherical.y;

    let x = r * sin(θ) * cos(φ);
    let y = r * sin(θ) * sin(φ);
    let z = r * cos(θ);

    return vec3<f32>(x, y, z);
}

fn pack_geo(elem: GeometryElement) -> PackedGeometry {
    let albedo_pack = pack4x8unorm(vec4<f32>(elem.albedo, 0.));

    let normal_spherical = normal_to_spherical(elem.normal);
    let normal_pack = pack2x16unorm(vec2<f32>(
        normal_spherical.x * FRAC_1_PI,
        (normal_spherical.y + PI) * FRAC_1_PI * 0.5,
    ));

    let variance_depth_pack = pack2x16unorm(vec2<f32>(
        elem.variance,
        elem.depth,
    ));

    let pos = vec3<u32>(
        bitcast<u32>(elem.position.x),
        bitcast<u32>(elem.position.y),
        bitcast<u32>(elem.position.z),
    );

    let object_index_pack = (elem.object_index & 0x00FFFFFFu) | select(0u, 0x80000000u, elem.was_invalidated);
    let distance = bitcast<u32>(elem.distance_from_origin);

    return PackedGeometry(
        vec4<u32>(
            albedo_pack,
            normal_pack,
            variance_depth_pack,
            pos.x,
        ), vec4<u32>(
            pos.y,
            pos.z,
            object_index_pack,
            bitcast<u32>(elem.similarity_score),
        )
    );
}

fn unpack_geo(geo: PackedGeometry) -> GeometryElement {
    let variance_depth = unpack2x16unorm(geo.pack_0[2]);
    let spherical_normal = unpack2x16unorm(geo.pack_0[1]);
    let position = vec3<f32>(
        bitcast<f32>(geo.pack_0[3]),
        bitcast<f32>(geo.pack_1[0]),
        bitcast<f32>(geo.pack_1[1]),
    );

    return GeometryElement(
        /* albedo   */ unpack4x8unorm(geo.pack_0[0]).xyz,
        /* variance */ variance_depth.x,
        /* normal   */ normal_from_spherical(vec2<f32>(
            spherical_normal.x * PI,
            (spherical_normal.y * 2. - 1.) * PI,
        )),
        /* depth    */ variance_depth.y,
        /* position */ position,
        /* distance */ length(position),
        /* index    */ geo.pack_1[2] & 0x00FFFFFF,
        /* inval'd  */ (geo.pack_1[2] & 0x80000000u) == 0x80000000u,
        /* s-lity   */ bitcast<f32>(geo.pack_1[3]),
    );
}

fn collect_geo_i(coords: vec2<i32>) -> GeometryElement {
    return collect_geo_u(vec2<u32>(max(coords, vec2<i32>(0))));
}

fn collect_geo_t2d(coords: vec2<u32>, pack_0: texture_2d<u32>, pack_1: texture_2d<u32>) -> GeometryElement {
    let sample_pack_0 = textureLoad(pack_0, coords, 0);
    let sample_pack_1 = textureLoad(pack_1, coords, 0);

    return unpack_geo(PackedGeometry(sample_pack_0, sample_pack_1));
}

fn collect_geo_ts2d(coords: vec2<u32>, pack_0: texture_storage_2d<rgba32uint, read_write>, pack_1: texture_storage_2d<rgba32uint, read_write>) -> GeometryElement {
    let sample_pack_0 = textureLoad(pack_0, coords);
    let sample_pack_1 = textureLoad(pack_1, coords);

    return unpack_geo(PackedGeometry(sample_pack_0, sample_pack_1));
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