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
fn rotl(v: u32, amt: u32) -> u32 {
    return (v << amt) | (v >> (32u - amt));
}

struct RNGState {
    state: array<u32, 4>,
}

fn rng_permute_impl_xoroshiro128(arr: array<u32, 4>) -> array<u32, 4> {
    var ret = arr;
    let t = ret[1] << 9u;

    ret[2] ^= ret[0];
    ret[3] ^= ret[1];
    ret[1] ^= ret[2];
    ret[0] ^= ret[3];

    ret[2] ^= t;
    ret[3] = rotl(ret[3], 11u);

    return ret;
}

fn rng_scramble_impl_xoroshiro128pp(s: array<u32, 4>) -> u32 {
    return rotl(s[0] + s[3], 7u) + s[0];
}

fn rng_permute_impl_xoroshiro64(arr: array<u32, 2>) -> array<u32, 2> {
    var ret = arr;
    
    let s1 = arr[0] ^ arr[1];
    ret[0] = rotl(arr[0], 26u) ^ s1 ^ (s1 << 9u);
    ret[1] = rotl(s1, 13u);

    return ret;
}

fn rng_scramble_impl_xoroshiro64s(s: array<u32, 2>) -> u32 {
    return s[0] * 0x9E3779BBu;
}

/*fn rng_next(state: ptr<function, RNGState>) -> u32 {
    // https://github.com/gfx-rs/wgpu/issues/4549
    //return rng_next_impl_xoroshiro128(&((*state).state));

    let ret = rng_scramble_impl_xoroshiro128pp((*state).state);
    (*state).state = rng_permute_impl_xoroshiro128((*state).state);
    return ret;
}*/

fn setup_rng_impl_xoroshiro128(pixel: vec2<u32>, dims: vec2<u32>, pix_hash: u32, pix_seed: vec4<u32>) -> array<u32, 4> {
    return array<u32, 4>(
        uniforms.seed_0 ^ pix_hash ^ pix_seed.r,
        uniforms.seed_1 ^ pix_hash ^ pix_seed.g,
        uniforms.seed_2 ^ pix_hash ^ pix_seed.b,
        uniforms.seed_3 ^ pix_hash ^ pix_seed.a
    );
}

fn setup_rng_impl_xoroshiro64(pixel: vec2<u32>, dims: vec2<u32>, pix_hash: u32, pix_seed: vec4<u32>) -> array<u32, 2> {
    return array<u32, 2>(
        uniforms.seed_0 ^ pix_hash ^ pix_seed.r,
        uniforms.seed_2 ^ pix_hash ^ pix_seed.b,
    );
}

var<private> rng_state: RNGState;
var<private> rng_mix_value: u32 = 0u;

fn rand() -> f32 {
    let base_res = rng_scramble_impl_xoroshiro128pp(rng_state.state);
    rng_state.state = rng_permute_impl_xoroshiro128(rng_state.state);

    let u32_val = base_res ^ rng_mix_value;

    return ldexp(f32(u32_val & 0xFFFFFFu), -24);
}

fn setup_rng(pixel: vec2<u32>, dims: vec2<u32>, local_idx: u32) {
    let pix_hash = pixel.x * 0x9e3779b9u ^ pixel.y * 0x517cc1b7u;

    let pix_seed = textureLoad(texture_noise, pixel % textureDimensions(texture_noise));
    rng_state = RNGState(setup_rng_impl_xoroshiro128(pixel, dims, pix_hash, pix_seed));

    // produced using the following three lines executed in the NodeJS REPL:
    // const crypto = await import("crypto");
    // let rand_u32=()=>"0x"+crypto.randomBytes(4).toString("hex")+'u'
    // [...Array.from(Array(8)).keys()].map(_=>[...Array.from(Array(8)).keys()].map(rand_u32).join(", "))
    // had to use "var" instead of "let" or "const" because of the non-uniform index "local_idx"
    var wg_schedule = array<u32, 64>(
        0x9452e4a1u, 0x3e12e3d1u, 0x59c57a43u, 0x03dad6d0u, 0x2e451baau, 0x46753a2bu, 0x2f95dae5u, 0xaa53a29cu,
        0xeb573daau, 0x5a7a5ebbu, 0xd072f2fcu, 0x235b9f9du, 0xc36cd687u, 0xfc250249u, 0x5bbed342u, 0xb2a788dfu,
        0xfa8d2fa0u, 0xbb778397u, 0xddfcdf2du, 0x7872fa4eu, 0x66540a17u, 0xea51b619u, 0xaaab34a5u, 0x4af4e53au,
        0x2e24a056u, 0xd552c708u, 0x645cae0au, 0xf67082dbu, 0x50d1e3bfu, 0x6ea3fed1u, 0x90ac2748u, 0xa079cd46u,
        0x5a831b23u, 0xc87fac2bu, 0x7b629adcu, 0xd966d8f1u, 0x30bfb83cu, 0xadb8a7dcu, 0xb2edab23u, 0xc2931362u,
        0x241fbd80u, 0xc1d86eedu, 0x4702d255u, 0x4cd45d07u, 0x4ffd0c09u, 0x8240bd19u, 0x5ac940e1u, 0x6ea39e5cu,
        0x9907e6d7u, 0x1d058790u, 0xa30de070u, 0x23946026u, 0x6e4accd2u, 0x5d35734du, 0x4489345bu, 0xf594bf72u,
        0x90bc2ef3u, 0x5e64c258u, 0x2fd0b938u, 0xd76fa8a6u, 0x53fb501cu, 0x53916405u, 0x0ccbf8a6u, 0x17067a8du
    );

    rng_mix_value = wg_schedule[local_idx % 64u] ^ pix_hash ^ pix_seed.x;
}
fn box_muller_map(unorm_samples: vec2<f32>) -> vec2<f32> {
    let r_2 = -2. * log(unorm_samples.x);
    let r = sqrt(r_2);
    let θ = 2. * PI * unorm_samples.y;
    return vec2<f32>(sin(θ), cos(θ)) * r;
}

fn u32_hash(v: u32) -> u32 {
    let x = ((v >> 16u) ^ v) * 0x45d9f3bu;
    let y = ((x >> 16u) ^ x) * 0x45d9f3bu;
    let z = ((y >> 16u) ^ y);

    return z;
}

fn vec2_u32_hash(v: vec2<u32>) -> u32 {
    return (53u + u32_hash(v.y)) * 53u + u32_hash(v.x);
}

fn u32_to_f32_unorm(v: u32) -> f32 {
    return ldexp(f32(v & 0xFFFFFFu), -24);
}

var<private> box_muller_cache: f32;
var<private> box_muller_cache_full: bool = false;

fn box_muller() -> f32 {
    box_muller_cache_full = !box_muller_cache_full;

    if !box_muller_cache_full {
        return box_muller_cache;
    }

    let samples = box_muller_map(vec2<f32>(
        rand(),
        rand(),
    ));

    box_muller_cache = samples.y;

    return samples.x;
}

fn sample_sphere_3d(out_probability: ptr<function, f32>) -> vec3<f32> {
    let norm_vec = vec3<f32>(
        box_muller_map(vec2<f32>(
            rand(),
            rand(),
        )),
        box_muller(),
    );

    *out_probability = 0.25 * FRAC_1_PI;

    return normalize(norm_vec);
}

fn sample_cos_hemisphere_3d(out_probability: ptr<function, f32>) -> vec3<f32> {
    let cosθ = sqrt(rand());
    let sinθ = sqrt(1. - cosθ * cosθ);
    *out_probability = cosθ * FRAC_1_PI;

    let φ = 2. * PI * rand();

    let sinφ = sin(φ);
    let cosφ = cos(φ);

    return vec3<f32>(cosφ * sinθ, sinφ * sinθ, cosθ);
}
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

fn orthonormal_from_xz(x: vec3<f32>, z: vec3<f32>) -> mat3x3<f32> {
    let y = cross(z, x);

    return mat3x3<f32>(
        x[0], x[1], x[2],
        y[0], y[1], y[2],
        z[0], z[1], z[2],
    );
}

// https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
fn rodrigues(x: vec3<f32>, norm_from: vec3<f32>, norm_to: vec3<f32>) -> vec3<f32> {
    let along = normalize(cross(norm_from, norm_to));
    let cosφ = dot(norm_from, norm_to);
    let sinφ = length(cross(norm_from, norm_to));
    if abs(cosφ) > 0.999 {
        return select(x, -x, cosφ < 0.);
    } else {
        return x * cosφ + cross(along, x) * sinφ + along * dot(along, x) * (1. - cosφ);
    }
}

// only for reflection space to surface space conversions
fn rodrigues_fast(x: vec3<f32>, norm_to: vec3<f32>) -> vec3<f32> {
    let cosφ = norm_to.z;
    let sinφ = sqrt(norm_to.y * norm_to.y + norm_to.x * norm_to.x);

    let along = vec3<f32>(-norm_to.y, norm_to.x, 0.) / sinφ;

    if abs(cosφ) > 0.999 {
        return select(x, -x, cosφ < 0.);
    } else {
        return x * cosφ + cross(along, x) * sinφ + along * dot(along, x) * (1. - cosφ);
    }
}

struct Intersection {
    distance: f32,
    position: vec3<f32>,
    normal: vec3<f32>,
    wo: vec3<f32>,
    material_idx: u32,

    //refl_to_surface: mat3x3<f32>,
}

fn dummy_intersection(ray: Ray) -> Intersection {
    return Intersection(INF, vec3<f32>(INF), -ray.direction, -ray.direction, 0u);
}

fn intersection_going_in(intersection: Intersection) -> bool { return 0. < dot(intersection.wo, intersection.normal); }
fn intersection_oriented_normal(intersection: Intersection) -> vec3<f32> { return select(-intersection.normal, intersection.normal, intersection_going_in(intersection)); }
fn intersection_cos_theta_o(intersection: Intersection) -> f32 { return abs(dot(intersection.wo, intersection.normal)); }

fn pick_intersection(lhs: Intersection, rhs: Intersection) -> Intersection {
    if lhs.distance > rhs.distance {
        return rhs;
    } else {
        return lhs;
    }
}

fn reflect(wo: vec3<f32>, oriented_normal: vec3<f32>) -> vec3<f32> {
    return -wo + oriented_normal * 2. * dot(oriented_normal, wo);
}

fn schlick(cosθ: f32, η1: f32, η2: f32) -> f32 {
    let sqrt_r0 = ((η1 - η2) / (η1 + η2));
    let r0 = sqrt_r0 * sqrt_r0;
    let r = r0 + (1. - r0) * pow(1. - cosθ, 5.);

    return r;
}

fn refract(
    wo: vec3<f32>,
    normal: vec3<f32>,
    incident_index: f32,
    transmittant_index: f32,
    out_probability_reflection: ptr<function, f32>,
    out_refraction: ptr<function, vec3<f32>>,
) -> bool {
    let l = -wo;

    let index_ratio = incident_index / transmittant_index;

    let cosθ_i = dot(-l, normal);
    let sin2θ_i = 1. - cosθ_i * cosθ_i;
    let sin2θ_t = index_ratio * index_ratio * sin2θ_i;

    if sin2θ_t >= 1. {
        return false;
    }

    let cosθ_t = sqrt(1. - sin2θ_t);

    *out_probability_reflection = schlick(cosθ_i, incident_index, transmittant_index);
    *out_refraction = l * index_ratio + normal * (index_ratio * cosθ_i - cosθ_t);

    return true;
}

struct SurfaceSample {
    position: vec3<f32>,
    uv: vec2<f32>,
    normal: vec3<f32>,
    pdf: f32,
}
struct PinpointCamera {
    fov: f32,
}

fn pinpoint_generate_ray(
    camera: PinpointCamera,
    screen_coords: vec2<u32>,
    screen_dims: vec2<u32>,
    pos: vec3<f32>,
) -> Ray {
    let half_theta = camera.fov / 2.;
    let d = (1. / (2. * sin(half_theta))) * sqrt(abs((f32(screen_dims.x) * (2. - f32(screen_dims.x)))));

    //let offset = vec2<f32>(rand() * 2. - 1., rand() * 2. - 1.);

    let direction = normalize(vec3<f32>(
        f32(screen_coords.x) - (f32(screen_dims.x) / 2.),
        f32(screen_dims.y - screen_coords.y - 1u) - f32(screen_dims.y) / 2.,
        d,
    ));

    return Ray(pos, direction);
}
struct Sphere {
    center: vec3<f32>,
    radius: f32,
    material: u32,
}

// value written to out is indeterminate if the function returns false.
fn solve_sphere_quadratic(a: f32, b: f32, c: f32, out: ptr<function, f32>) -> bool {
    let delta = b * b - 4. * a * c;
    let sqrt_delta = sqrt(delta);

    let raw_sol_0 = (-b + sqrt_delta) / (2. * a);
    let raw_sol_1 = (-b - sqrt_delta) / (2. * a);

    // nans should become infs here
    // i don't know actually, is nan ordered in wgsl? does wgsl even have a nan?
    let sol_0 = select(INF, raw_sol_0, raw_sol_0 >= 0.);
    let sol_1 = select(INF, raw_sol_1, raw_sol_1 >= 0.);

    let solution = select(sol_1, sol_0, sol_0 < sol_1);

    *out = solution;

    return delta >= 0.;
}

fn sphere_uv(local_point: vec3<f32>, radius: f32) -> vec2<f32> {
    let θ_uncorrected = atan2(local_point[1], local_point[0]);
    let θ = select(θ_uncorrected, θ_uncorrected + 2. * PI, θ_uncorrected < 0.);

    let φ = PI - acos(local_point[2] / radius);

    let u = θ * 0.5 * FRAC_1_PI;
    let v = φ * FRAC_1_PI;

    return vec2<f32>(u, v);
}

fn sphere_surface_params(local_point: vec3<f32>, radius: f32, uv: vec2<f32>) -> array<vec3<f32>, 2> {
    let π = PI;
    let x = local_point.x;
    let y = local_point.y;

    let θ = uv[0] * 2. * π;
    let φ = uv[1] * π;

    let sinθ = sin(θ);
    let cosθ = cos(θ);

    let δxδu = -2. * π * y;
    let δyδu = 2. * π * x;
    let δzδu = 0.;

    let δxδv = 2. * π * cosθ;
    let δyδv = 2. * π * sinθ;
    let δzδv = -radius * π * sin(φ);

    return array<vec3<f32>, 2>(
        vec3<f32>(δxδu, δyδu, δzδu),
        vec3<f32>(δxδv, δyδv, δzδv)
    );
}

fn sphere_intersect(sphere: Sphere, ray: Ray, best: f32, out: ptr<function, Intersection>) -> bool {
    let direction = ray.origin - sphere.center;

    let a = 1.;
    let b = 2. * dot(direction, ray.direction);
    let c = dot(direction, direction) - (sphere.radius * sphere.radius);

    var t = 0.;
    if !solve_sphere_quadratic(a, b, c, &t) {
        return false;
    }

    let global_position = ray.origin + ray.direction * t;
    let local_position = global_position - sphere.center;
    let normal = normalize(local_position);
    let oriented_normal = select(-normal, normal, dot(ray.direction, normal) < 0.);
    let uv = sphere_uv(local_position, sphere.radius);

    //let surface_params = sphere_surface_params(local_position, sphere.radius, uv);
    //let refl_to_surface = orthonormal_from_xz(normalize(surface_params[0]), oriented_normal);

    *out = Intersection(
        t,
        global_position,
        normal,
        -ray.direction,
        sphere.material,
        //refl_to_surface,
    );

    return true;
}

fn sphere_sample(sphere: Sphere) -> SurfaceSample {
    var pdf: f32;
    let normal = sample_sphere_3d(&pdf);
    pdf /= sphere.radius * sphere.radius;

    let local_position = normal * sphere.radius;
    let global_position = local_position + sphere.center;

    return SurfaceSample(
        global_position,
        sphere_uv(local_position, sphere.radius),
        normal,
        pdf,
    );
}
struct Triangle {
    vertices: array<vec4<f32>, 3>,
    material: u32,
    padding_0: u32,
    padding_1: u32,
    padding_2: u32,
}

// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
fn triangle_intersect(triangle: Triangle, ray: Ray, best: f32, out: ptr<function, Intersection>) -> bool {
    let eps = 0.0001;

    let edge1 = triangle.vertices[1].xyz - triangle.vertices[0].xyz;
    let edge2 = triangle.vertices[2].xyz - triangle.vertices[0].xyz;
    let ray_cross_e2 = cross(ray.direction, edge2);
    let det = dot(edge1, ray_cross_e2);

    if det > -eps && det < eps {
        return false;
    }

    let inv_det = 1.0 / det;
    let s = ray.origin - triangle.vertices[0].xyz;
    let u = inv_det * dot(s, ray_cross_e2);

    if u < 0. || u > 1. {
        return false;
    }

    let s_cross_e1 = cross(s, edge1);
    let v = inv_det * dot(ray.direction, s_cross_e1);

    if v < 0. || u + v > 1. {
        return false;
    }

    let t = inv_det * dot(edge2, s_cross_e1);

    if t < eps || best < t {
        return false;
    }

    let normal = normalize(cross(edge1, edge2));
    let oriented_normal = select(-normal, normal, dot(ray.direction, normal) < 0.);

    *out = Intersection(
        t,
        ray.origin + ray.direction * t,
        normal,
        -ray.direction,
        triangle.material,
        //orthonormal_from_xz(normalize(edge1), oriented_normal),
    );

    return true;
}

fn triangle_area(triangle: Triangle) -> f32 {
    let edge1 = triangle.vertices[1].xyz - triangle.vertices[0].xyz;
    let edge2 = triangle.vertices[2].xyz - triangle.vertices[0].xyz;
    let edge_cross = cross(edge1, edge2);
    return length(edge_cross) / 2.;
}

fn triangle_sample(triangle: Triangle) -> SurfaceSample {
    let uv_square = vec2<f32>(rand(), rand());
    let uv_folded = vec2<f32>(1. - uv_square.y, 1. - uv_square.x);
    let upper_right = uv_square.x + uv_square.y > 1.;
    let uv = select(uv_square, uv_folded, upper_right);

    let edge1 = triangle.vertices[1].xyz - triangle.vertices[0].xyz;
    let edge2 = triangle.vertices[2].xyz - triangle.vertices[0].xyz;
    let edge_cross = cross(edge1, edge2);
    let double_area = length(edge_cross);
    let normal = edge_cross / double_area;

    let pt = triangle.vertices[0].xyz + edge1 * uv.x + edge2 * uv.y;

    return SurfaceSample(
        pt,
        uv,
        normal,
        2. / double_area,
    );
}
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
@group(0) @binding(1) var texture_noise: texture_storage_2d<rgba32uint, read>;

@group(1) @binding(0) var texture_rt: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(1) var geo_texture_albedo: texture_2d<f32>;
@group(1) @binding(2) var geo_texture_pack_normal_depth: texture_2d<f32>;
@group(1) @binding(3) var geo_texture_pack_pos_dist: texture_2d<f32>;
@group(1) @binding(4) var geo_texture_object_index: texture_2d<u32>;
@group(1) @binding(5) var texture_denoise_0: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(6) var texture_denoise_1: texture_storage_2d<rgba8unorm, read_write>;

@group(2) @binding(0) var<storage> triangles: array<Triangle>;

struct Material {
    albedo: vec3<f32>,
    emittance: vec3<f32>,
    // 0 -> diffuse, 1 -> perfect mirror, 2 -> dielectric, 3 -> glossy (NYI)
    mat_type: u32,
    glossiness: f32,
    index: f32,
}

var<private> materials: array<Material, 9> = array<Material, 9>(
    Material(vec3<f32>(1.)  , vec3<f32>(12.), 0u, 0., 1. ),         // 0 light
    Material(vec3<f32>(0.99), vec3<f32>(0.) , 1u, 1., 1. ),         // 1 mirror
    Material(vec3<f32>(0.99), vec3<f32>(0.) , 2u, 0., 1.5),         // 2 glass
    Material(vec3<f32>(0.75), vec3<f32>(0.) , 0u, 0., 1. ),         // 3 white
    Material(vec3<f32>(0.75, 0.25, 0.25), vec3<f32>(0.), 0u, 0., 1.), // 4 red
    Material(vec3<f32>(0.25, 0.25, 0.75), vec3<f32>(0.), 0u, 0., 1.), // 5 blue
    Material(vec3<f32>(1., 0., 0.), vec3<f32>(12., 0., 0.), 0u, 0., 1.), // 6 light (red)
    Material(vec3<f32>(0., 1., 0.), vec3<f32>(0., 12., 0.), 0u, 0., 1.), // 7 light (green)
    Material(vec3<f32>(0., 0., 1.), vec3<f32>(0., 0., 12.), 0u, 0., 1.), // 8 light (blyue)
);

const CBL: vec3<f32> = vec3<f32>(-3.5, -3.5, -20.);
const CTR: vec3<f32> = vec3<f32>(3.5, 2.5, 20.);

/*
const NUM_TRIANGLES = 30u;
var<private> triangles: array<Triangle, NUM_TRIANGLES> = array<Triangle, NUM_TRIANGLES>(
    // light 1
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-3., 2.4, 15.), vec3<f32>(-1., 2.4, 15.), vec3<f32>(-1., 2.4, 11.25)), 6u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-1., 2.4, 11.25), vec3<f32>(-3., 2.4, 11.25), vec3<f32>(-3., 2.4, 15.)), 6u),

    // light 2
    Triangle(array<vec3<f32>, 3>(vec3<f32>(1., 2.4, 15.), vec3<f32>(3., 2.4, 15.), vec3<f32>(3., 2.4, 11.25)), 7u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(3., 2.4, 11.25), vec3<f32>(1., 2.4, 11.25), vec3<f32>(1., 2.4, 15.)), 7u),

    // light 3
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-1.25, 2.4, 12.), vec3<f32>(1.25, 2.4, 12.), vec3<f32>(1.25, 2.4, 8.25)), 8u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(1.25, 2.4, 8.25), vec3<f32>(-1.25, 2.4, 8.25), vec3<f32>(-1.25, 2.4, 12.)), 8u),

    // light 2
    //Triangle(array<vec3<f32>, 3>(vec3<f32>(-1.25, 2.4, 15.), vec3<f32>(1.25, 2.4, 15.), vec3<f32>(1.25, 2.4, 11.25)), 0u),
    //Triangle(array<vec3<f32>, 3>(vec3<f32>(1.25, 2.4, 11.25), vec3<f32>(-1.25, 2.4, 11.25), vec3<f32>(-1.25, 2.4, 15.)), 0u),

    // mirror prism (bounding box: [-2.65, -2.5, 16.6], [-0.85, -0.7, 18.4])
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-2.65, -2.5, 16.6),
        vec3<f32>(-2.65, -2.5, 18.4),
        vec3<f32>(-0.85, -2.5, 18.4),
    ), 1u), // bottom 1
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-0.85, -2.5, 18.4),
        vec3<f32>(-0.85, -2.5, 16.6),
        vec3<f32>(-2.65, -2.5, 16.6),
    ), 1u), // bottom 2
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-2.65, -2.5, 18.4),
        vec3<f32>(-2.65, -2.5, 16.6),
        vec3<f32>((-2.65 + -0.85) / 2., -0.7, (16.6 + 18.4) / 2.),
    ), 1u), // west
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-2.65, -2.5, 16.6),
        vec3<f32>(-0.85, -2.5, 16.6),
        vec3<f32>((-2.65 + -0.85) / 2., -0.7, (16.6 + 18.4) / 2.),
    ), 1u), // south
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-0.85, -2.5, 16.6),
        vec3<f32>(-0.85, -2.5, 18.4),
        vec3<f32>((-2.65 + -0.85) / 2., -0.7, (16.6 + 18.4) / 2.),
    ), 1u), // east
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-0.85, -2.5, 18.4),
        vec3<f32>(-2.65, -2.5, 18.4),
        vec3<f32>((-2.65 + -0.85) / 2., -0.7, (16.6 + 18.4) / 2.),
    ), 1u), // north

    // glass prism (bounding box: [0.85, -2.3, 15.6], [2.65, -0.5, 17.4])
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-2.65, -2.5, 16.6) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>(-2.65, -2.5, 18.4) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>(-0.85, -2.5, 18.4) + vec3<f32>(3.5, 0.2, -1.),
    ), 2u), // bottom 1
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-0.85, -2.5, 18.4) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>(-0.85, -2.5, 16.6) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>(-2.65, -2.5, 16.6) + vec3<f32>(3.5, 0.2, -1.),
    ), 2u), // bottom 2
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-2.65, -2.5, 18.4) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>(-2.65, -2.5, 16.6) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>((-2.65 + -0.85) / 2., -0.7, (16.6 + 18.4) / 2.) + vec3<f32>(3.5, 0.2, -1.) + vec3<f32>(0., -0.3, 0.),
    ), 2u), // west
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-2.65, -2.5, 16.6) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>(-0.85, -2.5, 16.6) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>((-2.65 + -0.85) / 2., -0.7, (16.6 + 18.4) / 2.) + vec3<f32>(3.5, 0.2, -1.) + vec3<f32>(0., -0.3, 0.),
    ), 2u), // south
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-0.85, -2.5, 16.6) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>(-0.85, -2.5, 18.4) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>((-2.65 + -0.85) / 2., -0.7, (16.6 + 18.4) / 2.) + vec3<f32>(3.5, 0.2, -1.) + vec3<f32>(0., -0.3, 0.),
    ), 2u), // east
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-0.85, -2.5, 18.4) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>(-2.65, -2.5, 18.4) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>((-2.65 + -0.85) / 2., -0.7, (16.6 + 18.4) / 2.) + vec3<f32>(3.5, 0.2, -1.) + vec3<f32>(0., -0.3, 0.),
    ), 2u), // north

    // front wall
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CTR.y, CTR.z), vec3<f32>(CBL.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CTR.z)), 3u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CTR.y, CTR.z), vec3<f32>(CBL.x, CTR.y, CTR.z)), 3u),

    // back wall
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CBL.y, CBL.z), vec3<f32>(CTR.x, CBL.y, CBL.z)), 3u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CBL.y, CBL.z), vec3<f32>(CTR.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CBL.z)), 3u),

    // right wall
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CTR.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CBL.z)), 5u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CBL.y, CBL.z), vec3<f32>(CTR.x, CTR.y, CBL.z), vec3<f32>(CTR.x, CTR.y, CTR.z)), 5u),

    // ceiling
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CTR.y, CTR.z), vec3<f32>(CTR.x, CTR.y, CTR.z), vec3<f32>(CTR.x, CTR.y, CBL.z)), 3u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CTR.z)), 3u),

    // left wall
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CTR.y, CTR.z), vec3<f32>(CBL.x, CBL.y, CTR.z), vec3<f32>(CBL.x, CBL.y, CBL.z)), 4u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CBL.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CTR.z)), 4u),

    // floor
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CBL.z)), 3u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CBL.y, CBL.z), vec3<f32>(CBL.x, CBL.y, CBL.z), vec3<f32>(CBL.x, CBL.y, CTR.z)), 3u),
);
*/

const NUM_EMISSIVE: u32 = 2u;
var<private> emissive_triangles: array<u32, 6> = array<u32, 6>(0u, 1u, 2u, 3u, 4u, 5u);

fn intersect_stuff(ray: Ray, out_intersection: ptr<function, Intersection>) -> bool {
    var intersected = false;

    for (var i = 0u; i < arrayLength(&triangles); i++) {
        if triangle_intersect(triangles[i], ray, (*out_intersection).distance, out_intersection) {
            intersected = true;
        }
    }

    return intersected;
}

fn sample_direct_lighting(intersection: Intersection) -> vec3<f32> {
    let material = materials[intersection.material_idx];
    
    if material.mat_type != 0u {
        return vec3<f32>(0.);
    }

    var sum = vec3<f32>(0.);

    let going_in = dot(intersection.wo, intersection.normal) > 0.;
    let oriented_normal = select(-intersection.normal, intersection.normal, going_in);

    for (var i = 0u; i < NUM_EMISSIVE; i++) {
        let tri_idx = emissive_triangles[i];
        let tri = triangles[tri_idx];
        let sample = triangle_sample(tri);
        let material = materials[tri.material];

        let vector_towards_light = sample.position - intersection.position;
        let square_distance = dot(vector_towards_light, vector_towards_light);
        let distance_to_light = sqrt(square_distance);
        let wi = vector_towards_light / distance_to_light;

        let hitcheck_ray = Ray(sample.position, wi);
        var hitcheck_intersection: Intersection;
        if intersect_stuff(hitcheck_ray, &hitcheck_intersection)
            && abs(hitcheck_intersection.distance - distance_to_light) > 0.01 {
            continue;
        }

        let brdf = FRAC_1_PI * 0.5;
        //let power_heuristic = (sample.pdf * sample.pdf) / (sample.pdf * sample.pdf + brdf * brdf);

        let p = abs(dot(sample.normal, wi)) / dot(vector_towards_light, vector_towards_light);
        sum += material.emittance * abs(dot(intersection.normal, vector_towards_light)) * p / triangle_area(tri);
    }

    return sum;
}

fn get_wi_and_weight(intersection: Intersection, out_was_specular: ptr<function, bool>) -> vec4<f32> {
    let material = materials[intersection.material_idx];
    let going_in = dot(intersection.wo, intersection.normal) > 0.;
    let oriented_normal = select(-intersection.normal, intersection.normal, going_in);

    var wi: vec3<f32>;
    var cos_brdf_over_wi_pdf: f32;
    var was_specular = false;

    switch material.mat_type {
        case 0u: {
            let brdf = FRAC_1_PI * 0.5;

            var sample_probability: f32;

            let importance_sample = sample_cos_hemisphere_3d(&sample_probability);
            wi = rodrigues_fast(importance_sample, oriented_normal);

            /*let sample = sample_sphere_3d(&sample_probability);
            wi = select(sample, -sample, dot(oriented_normal, sample) < 0.);
            sample_probability *= 2.;*/

            cos_brdf_over_wi_pdf = dot(wi, oriented_normal) * brdf / sample_probability;
        }

        case 1u: {
            was_specular = true;
            cos_brdf_over_wi_pdf = 1.;

            wi = reflect(intersection.wo, oriented_normal);
        }

        case 2u: {
            was_specular = true;
            cos_brdf_over_wi_pdf = 1.;

            let transmittant_index = select(1., material.index, going_in);
            let incident_index     = select(material.index, 1., going_in);

            var refraction: vec3<f32>;
            var p_reflection: f32;
            if refract(
                intersection.wo, oriented_normal,
                incident_index, transmittant_index,
                &p_reflection, &refraction
            ) {
                let generated = rand();

                if generated > p_reflection {
                    wi = refraction;
                } else {
                    wi = reflect(intersection.wo, oriented_normal);
                }
            } else {
                wi = reflect(intersection.wo, oriented_normal);
            }
        }

        default: {}
    }

    *out_was_specular = was_specular;
    return vec4<f32>(wi, cos_brdf_over_wi_pdf);
}

struct PixelSample {
    rt: vec3<f32>,
    direct_illum: vec3<f32>,
}

fn pixel_sample_add(lhs: PixelSample, rhs: PixelSample) -> PixelSample {
    return PixelSample(
        lhs.rt + rhs.rt,
        lhs.direct_illum + rhs.direct_illum,
    );
}

fn pixel_sample_div(lhs: PixelSample, divisor: f32) -> PixelSample {
    return PixelSample(
        lhs.rt / divisor,
        lhs.direct_illum / divisor,
    );
}

fn radiance(initial_intersection: Intersection) -> vec3<f32> {
    var attenuation = vec3<f32>(1.);
    var light = vec3<f32>(0.);

    var ray: Ray;
    {
        var _was_specular: bool;
        let wi_and_weight = get_wi_and_weight(initial_intersection, &_was_specular);
        let wi = wi_and_weight.xyz;
        let cos_brdf_over_wi_pdf = wi_and_weight.w;
        let offset = initial_intersection.normal * 0.0009 * select(1., -1., dot(wi, initial_intersection.normal) < 0.);
        ray = Ray(initial_intersection.position + offset, wi);
    }

    for (var i = 0u; i < 4u; i++) {
        var intersection: Intersection = dummy_intersection(ray);
        if !intersect_stuff(ray, &intersection) {
            break;
        }
        let material = materials[intersection.material_idx];

        // let explicit_lighting = sample_direct_lighting(intersection);

        let going_in = dot(ray.direction, intersection.normal) < 0.;
        let oriented_normal = select(-intersection.normal, intersection.normal, going_in);

        var _was_specular: bool;
        let wi_and_weight = get_wi_and_weight(intersection, &_was_specular);
        let wi = wi_and_weight.xyz;
        let cos_brdf_over_wi_pdf = wi_and_weight.w;

        let cur_attenuation = material.albedo * cos_brdf_over_wi_pdf;

        ray = Ray(intersection.position + intersection.normal * 0.0009 * select(1., -1., dot(wi, intersection.normal) < 0.), wi);

        light = light + (material.emittance * attenuation);
        attenuation = cur_attenuation * attenuation;
    }

    return light;
}

fn new_cs(pixel: vec2<u32>, dimensions: vec2<u32>) -> PixelSample {
    let geo_sample = collect_geo_u(pixel);
    var sample_info: PixelSample;

    var light = vec3<f32>(0.);
    var attenuation = vec3<f32>(1.);
    var hit_diffuse = false;

    var intersection = Intersection(
        length(geo_sample.position - uniforms.camera_position),
        geo_sample.position,
        geo_sample.normal,
        -normalize(geo_sample.position - uniforms.camera_position),
        triangles[geo_sample.object_index].material,
    );

    //intersection.position += intersection_oriented_normal(intersection) * 1;

    for (var depth = 0; depth < 4; depth++) {
        let material = materials[intersection.material_idx];

        var was_specular = false;
        let wi_and_weight = get_wi_and_weight(intersection, &was_specular);
        let wi = wi_and_weight.xyz;
        let cos_brdf_over_wi_pdf = wi_and_weight.w;

        light += attenuation * material.emittance;

        if depth >= 1 && !hit_diffuse && !was_specular {
            hit_diffuse = true;
            light += material.albedo * attenuation * sample_direct_lighting(intersection);
        }

        if depth == 0 {
            attenuation *= cos_brdf_over_wi_pdf;
        } else {
            attenuation *= material.albedo * cos_brdf_over_wi_pdf;
        }

        let offset = intersection.normal * 0.009 * select(1., -1., dot(wi, intersection.normal) < 0.);
        let ray = Ray(intersection.position + offset, wi);
        var new_intersection: Intersection = dummy_intersection(ray);
        if !intersect_stuff(ray, &new_intersection) {
            break;
        }
        intersection = new_intersection;
    }

    light += attenuation * materials[intersection.material_idx].emittance;

    sample_info.rt = light;

    return sample_info;
}

fn check_discrapency(pixel: vec2<u32>, dimensions: vec2<u32>) -> PixelSample {
    let geo_sample = collect_geo_u(pixel);

    var initial_intersection = Intersection(
        length(geo_sample.position - uniforms.camera_position),
        geo_sample.position,
        geo_sample.normal,
        -normalize(geo_sample.position - uniforms.camera_position),
        triangles[geo_sample.object_index].material,
    );
    
    let camera = PinpointCamera(FRAC_PI_4);
    //var ray = pinpoint_generate_ray(camera, pixel, dimensions, uniforms.camera_position);
    let direction = normalize(geo_sample.position - uniforms.camera_position);
    var ray = Ray(uniforms.camera_position, direction);

    var intersection: Intersection = dummy_intersection(ray);
    if !intersect_stuff(ray, &intersection) {
        return PixelSample(vec3<f32>(1., 0., 0.), vec3<f32>(0.));
    }

    //return PixelSample(vec3<f32>(initial_intersection.distance - intersection.distance), vec3<f32>(0.));
    //return PixelSample(vec3<f32>(length(initial_intersection.position - intersection.position) * 10.), vec3<f32>(0.));
    //return PixelSample(vec3<f32>(length(initial_intersection.normal - intersection.normal)), vec3<f32>(0.));
    //return PixelSample(vec3<f32>(length(initial_intersection.wo - intersection.wo)), vec3<f32>(0.));
    return PixelSample(vec3<f32>(f32(initial_intersection.material_idx - intersection.material_idx)), vec3<f32>(0.));
}

fn geo_and_rt(pixel: vec2<u32>, dimensions: vec2<u32>) -> PixelSample {
    let geo_sample = collect_geo_u(pixel);
    let direction = normalize(geo_sample.position - uniforms.camera_position);

    var sample_info: PixelSample;
    var hit_diffuse = false;    
    var ray = Ray(uniforms.camera_position, direction);

    var light = vec3<f32>(0.);
    var attenuation = vec3<f32>(1.);

    for (var depth = 0; depth < 5; depth++) {
        var intersection: Intersection = dummy_intersection(ray);
        if !intersect_stuff(ray, &intersection) {
            break;
        }
        let material = materials[intersection.material_idx];

        var was_specular = false;
        let wi_and_weight = get_wi_and_weight(intersection, &was_specular);
        let wi = wi_and_weight.xyz;
        let cos_brdf_over_wi_pdf = wi_and_weight.w;

        light += attenuation * material.emittance;

        if depth == 0 {
            attenuation *= cos_brdf_over_wi_pdf;
        } else {
            attenuation *= material.albedo * cos_brdf_over_wi_pdf;
        }

        // let offset = intersection_oriented_normal(intersection) * 0.0009;
        let offset = intersection.normal * 0.0009 * select(1., -1., dot(wi, intersection.normal) < 0.);
        ray = Ray(intersection.position + offset, wi);
    }

    sample_info.rt = light;

    return sample_info;
}

fn a_trous(
    tex_coords: vec2<i32>, tex_dims: vec2<i32>, step_scale: i32,
    tex_from: texture_storage_2d<rgba8unorm, read_write>,
) -> vec3<f32> {
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

    let center_rt = textureLoad(tex_from, tex_coords).xyz;
    let center_geo = collect_geo_i(tex_coords);

    var sum = vec3<f32>(0.);
    var kernel_sum = 0.;

    let σ_rt = 0.5;
    let σ_n  = 0.5;
    let σ_p  = 0.7;

    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            //let kernel_weight = kernel[(x + 2) + ((y + 2) * 5)];
            let kernel_weight = kernel[2 - clamp(2 - abs(x), 0, 2 - abs(y))];

            let offset = vec2<i32>(x, y) * step_scale;
            let cur_coords = clamp(tex_coords + offset, vec2<i32>(0), tex_dims);

            let sample_rt = textureLoad(tex_from, cur_coords).xyz;
            let dist_rt = distance(center_rt, sample_rt);
            let weight_rt = min(exp(-dist_rt / (σ_rt * σ_rt)), 1.);

            let sample_normal = collect_geo_i(cur_coords).normal;
            let dist_normal = distance(center_geo.normal, sample_normal);
            let weight_normal = min(exp(-dist_normal / (σ_n * σ_n)), 1.);

            /*let sample_pos = ge_position(geometry_buffer[gb_idx_i(cur_coords)]);
            let dist_pos = distance(ge_position(center_geo), sample_pos);
            let weight_pos = min(exp(-dist_pos / (σ_p * σ_p)), 1.);*/

            let sample_distance = collect_geo_i(cur_coords).distance_from_origin;
            let dist_distance = abs(sample_distance - center_geo.distance_from_origin);
            let weight_distance = min(exp(-dist_distance / (σ_p * σ_p)), 1.);

            let weight = kernel_weight * weight_rt * weight_normal * weight_distance;

            sum += weight * sample_rt.xyz;
            kernel_sum += weight;
        }
    }

    return sum / kernel_sum;
}

fn denoise_from_rt(pixel: vec2<u32>, texture_dimensions: vec2<u32>, stride: i32) {
    storageBarrier();
    textureStore(texture_denoise_0, pixel, vec4<f32>(a_trous(vec2<i32>(pixel), vec2<i32>(texture_dimensions), stride, texture_rt), 1.));
}

fn denoise_from_d0(pixel: vec2<u32>, texture_dimensions: vec2<u32>, stride: i32) {
    storageBarrier();
    textureStore(texture_denoise_1, pixel, vec4<f32>(a_trous(vec2<i32>(pixel), vec2<i32>(texture_dimensions), stride, texture_denoise_0), 1.));

}

fn denoise_from_d1(pixel: vec2<u32>, texture_dimensions: vec2<u32>, stride: i32) {
    storageBarrier();
    textureStore(texture_denoise_0, pixel, vec4<f32>(a_trous(vec2<i32>(pixel), vec2<i32>(texture_dimensions), stride, texture_denoise_1), 1.));
}

@compute @workgroup_size(8, 8) fn cs_main(
    @builtin(global_invocation_id)   global_id: vec3<u32>,
    @builtin(workgroup_id)           workgroup_id: vec3<u32>,
    @builtin(local_invocation_id)    local_id:  vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let pixel = global_id.xy;
    let texture_dimensions = textureDimensions(texture_rt);

    setup_rng(global_id.xy, texture_dimensions, local_idx);

    var sample_sum: PixelSample;
    let samples = 16;
    for (var i = 0; i < samples; i++) {
        //let sample = check_discrapency(pixel, texture_dimensions);
        //let sample = geo_and_rt(pixel, texture_dimensions);
        let sample = new_cs(pixel, texture_dimensions);
        sample_sum = pixel_sample_add(sample_sum, sample);
    }
    let sample = pixel_sample_div(sample_sum, f32(samples));

    textureStore(texture_rt, pixel, vec4<f32>(sample.rt, 1.));

    let geo_sample = collect_geo_u(pixel);
    let tri = triangles[geo_sample.object_index];
    let material = materials[tri.material];

    //textureStore(texture_rt, pixel, vec4<f32>(tri.vertices[0].xyz, 1.));
    //textureStore(texture_rt, pixel, vec4<f32>(material.emittance, 1.));

    denoise_from_rt(pixel, texture_dimensions, 1);
    denoise_from_d0(pixel, texture_dimensions, 2);
    denoise_from_d1(pixel, texture_dimensions, 3);
}
