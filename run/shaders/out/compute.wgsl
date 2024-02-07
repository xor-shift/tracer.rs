struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

struct MainUniform {
    // at frame no 0, texture 1 should be used and texture 0 should be drawn on
    frame_no: u32,
    current_instant: f32,
    seed_0: u32,
    seed_1: u32,
    seed_2: u32,
    seed_3: u32,
    visualisation_mode: i32,
}

struct GeometryElement {
    normal_and_depth: vec4<f32>,
    albedo: vec4<f32>,
    position: vec4<f32>,
}

fn gb_idx_i(coords: vec2<i32>) -> i32 {
    let cols = textureDimensions(texture_rt).x;
    return coords.x + coords.y * i32(cols);
}

fn gb_idx_u(coords: vec2<u32>) -> u32 {
    let cols = textureDimensions(texture_rt).x;
    return coords.x + coords.y * cols;
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
    let pix_hash = vec2_u32_hash(pixel);

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

    rng_mix_value = wg_schedule[local_idx % 64u] ^ pix_hash;
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

struct Intersection {
    distance: f32,
    position: vec3<f32>,
    normal: vec3<f32>,
    wo: vec3<f32>,
    material_idx: u32,

    refl_to_surface: mat3x3<f32>,
}

fn dummy_intersection(ray: Ray) -> Intersection {
    let inf = 1. / 0.;
    return Intersection(inf, vec3<f32>(inf), -ray.direction, -ray.direction, 0u, mat3x3<f32>(1., 0., 0., 0., 1., 0., 0., 0., 1.));
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
    oriented_normal: vec3<f32>,
    incident_index: f32,
    transmittant_index: f32,
    out_probability: ptr<function, f32>,
    out_refraction: ptr<function, vec3<f32>>,
) -> bool {
    let l = -wo;

    let index_ratio = incident_index / transmittant_index;

    let cosθ_i = dot(-l, oriented_normal);
    let sin2θ_i = 1. - cosθ_i * cosθ_i;
    let sin2θ_t = index_ratio * index_ratio * sin2θ_i;

    if sin2θ_t >= 1. {
        return false;
    }

    let cosθ_t = sqrt(1. - sin2θ_t);

    *out_probability = schlick(cosθ_i, incident_index, transmittant_index);
    *out_refraction = l * index_ratio + oriented_normal * (index_ratio * cosθ_i - cosθ_t);

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

    let offset = vec2<f32>(rand() * 2. - 1., rand() * 2. - 1.);

    let direction = normalize(vec3<f32>(
        f32(screen_coords.x) - (f32(screen_dims.x) / 2.) + offset.x,
        f32(screen_dims.y - screen_coords.y - 1u) - f32(screen_dims.y) / 2. + offset.y,
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
    let inf = 1. / 0.;
    let sol_0 = select(inf, raw_sol_0, raw_sol_0 >= 0.);
    let sol_1 = select(inf, raw_sol_1, raw_sol_1 >= 0.);

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

    let surface_params = sphere_surface_params(local_position, sphere.radius, uv);
    let refl_to_surface = orthonormal_from_xz(normalize(surface_params[0]), oriented_normal);

    *out = Intersection(
        t,
        global_position,
        normal,
        -ray.direction,
        sphere.material,
        refl_to_surface,
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
    vertices: array<vec3<f32>, 3>,
    uv_offset: vec2<f32>,
    uv_scale: vec2<f32>,
    material: u32,
}

// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
fn triangle_intersect(triangle: Triangle, ray: Ray, best: f32, out: ptr<function, Intersection>) -> bool {
    let eps = 0.0001;

    let edge1 = triangle.vertices[1] - triangle.vertices[0];
    let edge2 = triangle.vertices[2] - triangle.vertices[0];
    let ray_cross_e2 = cross(ray.direction, edge2);
    let det = dot(edge1, ray_cross_e2);

    if det > -eps && det < eps {
        return false;
    }

    let inv_det = 1.0 / det;
    let s = ray.origin - triangle.vertices[0];
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
        orthonormal_from_xz(normalize(edge1), oriented_normal),
    );

    return true;
}

fn triangle_sample(triangle: Triangle) -> SurfaceSample {
    let uv_square = vec2<f32>(rand(), rand());
    let uv_folded = vec2<f32>(1. - uv_square.y, 1. - uv_square.x);
    let upper_right = uv_square.x + uv_square.y > 1.;
    let uv = select(uv_square, uv_folded, upper_right);

    let edge1 = triangle.vertices[1] - triangle.vertices[0];
    let edge2 = triangle.vertices[2] - triangle.vertices[0];
    let edge_cross = cross(edge1, edge2);
    let double_area = length(edge_cross);
    let normal = edge_cross / double_area;

    let pt = triangle.vertices[0] + edge1 * uv.x + edge2 * uv.y;

    return SurfaceSample(
        pt,
        uv,
        normal,
        2. / double_area,
    );
}
@group(0) @binding(0) var<uniform> uniforms: MainUniform;
@group(0) @binding(1) var texture_noise: texture_storage_2d<rgba32uint, read>;

@group(1) @binding(0) var texture_rt: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(1) var<storage, read_write> geometry_buffer: array<GeometryElement>;
@group(1) @binding(2) var texture_denoise_0: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(3) var texture_denoise_1: texture_storage_2d<rgba8unorm, read_write>;

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
    Material(vec3<f32>(0.99), vec3<f32>(0.) , 2u, 0., 1.7),         // 2 glass
    Material(vec3<f32>(0.75), vec3<f32>(0.) , 0u, 0., 1. ),         // 3 white
    Material(vec3<f32>(0.75, 0.25, 0.25), vec3<f32>(0.), 0u, 0., 1.), // 4 red
    Material(vec3<f32>(0.25, 0.25, 0.75), vec3<f32>(0.), 0u, 0., 1.), // 5 blue
    Material(vec3<f32>(0.), vec3<f32>(12., 0., 0.), 0u, 0., 1.), // 6 light (red)
    Material(vec3<f32>(0.), vec3<f32>(0., 12., 0.), 0u, 0., 1.), // 7 light (green)
    Material(vec3<f32>(0.), vec3<f32>(0., 0., 12.), 0u, 0., 1.), // 8 light (blyue)
);

const NUM_SPHERES: u32 = 2u;
var<private> spheres: array<Sphere, NUM_SPHERES> = array<Sphere, 2>(
    Sphere(vec3<f32>(-1.75 , -2.5 + 0.9      , 17.5  ), .9     , 1u), // mirror
    Sphere(vec3<f32>(1.75  , -2.5 + 0.9 + 0.2, 16.5  ), .9     , 2u), // glass
    //Sphere(vec3<f32>(0.    , 42.499          , 15.   ), 40.    , 0u), // light
    /*
    Sphere(vec3<f32>(0.    , 0.              , -5000.), 4980.  , 3u), // front wall
    Sphere(vec3<f32>(0.    , 0.              , 5000. ), 4980.  , 3u), // backwall
    Sphere(vec3<f32>(5000. , 0.              , 0.    ), 4996.5 , 5u), // right wall
    Sphere(vec3<f32>(0.    , 5000.           , 5.    ), 4997.5 , 3u), // ceiling
    Sphere(vec3<f32>(-5000., 0.              , 0.    ), 4996.5 , 4u), // left wall
    Sphere(vec3<f32>(0.    , -5000.          , 5.    ), 4997.5 , 3u), // floor
    */
);

const CBL: vec3<f32> = vec3<f32>(-3.5, -2.5, -20.);
const CTR: vec3<f32> = vec3<f32>(3.5, 2.5, 20.);

const NUM_TRIANGLES = 14u;
var<private> triangles: array<Triangle, NUM_TRIANGLES> = array<Triangle, NUM_TRIANGLES>(
    // front wall
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CTR.y, CTR.z), vec3<f32>(CBL.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CTR.z)), vec2<f32>(0.), vec2<f32>(1.), 3u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CTR.y, CTR.z), vec3<f32>(CBL.x, CTR.y, CTR.z)), vec2<f32>(0.), vec2<f32>(1.), 3u),

    // back wall
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CBL.y, CBL.z), vec3<f32>(CTR.x, CBL.y, CBL.z)), vec2<f32>(0.), vec2<f32>(1.), 3u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CBL.y, CBL.z), vec3<f32>(CTR.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CBL.z)), vec2<f32>(0.), vec2<f32>(1.), 3u),

    // right wall
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CTR.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CBL.z)), vec2<f32>(0.), vec2<f32>(1.), 5u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CBL.y, CBL.z), vec3<f32>(CTR.x, CTR.y, CBL.z), vec3<f32>(CTR.x, CTR.y, CTR.z)), vec2<f32>(0.), vec2<f32>(1.), 5u),

    // ceiling
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CTR.y, CTR.z), vec3<f32>(CTR.x, CTR.y, CTR.z), vec3<f32>(CTR.x, CTR.y, CBL.z)), vec2<f32>(0.), vec2<f32>(1.), 3u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CTR.z)), vec2<f32>(0.), vec2<f32>(1.), 3u),

    // left wall
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CTR.y, CTR.z), vec3<f32>(CBL.x, CBL.y, CTR.z), vec3<f32>(CBL.x, CBL.y, CBL.z)), vec2<f32>(0.), vec2<f32>(1.), 4u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CBL.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CTR.z)), vec2<f32>(0.), vec2<f32>(1.), 4u),

    // floor
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CBL.z)), vec2<f32>(0.), vec2<f32>(1.), 3u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CBL.y, CBL.z), vec3<f32>(CBL.x, CBL.y, CBL.z), vec3<f32>(CBL.x, CBL.y, CTR.z)), vec2<f32>(0.), vec2<f32>(1.), 3u),

    /*
    // light 1
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-3., 2.4, 15.), vec3<f32>(-1., 2.4, 15.), vec3<f32>(-1., 2.4, 11.25)), vec2<f32>(0.), vec2<f32>(1.), 6u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-1., 2.4, 11.25), vec3<f32>(-3., 2.4, 11.25), vec3<f32>(-3., 2.4, 15.)), vec2<f32>(0.), vec2<f32>(1.), 6u),

    // light 2
    Triangle(array<vec3<f32>, 3>(vec3<f32>(1., 2.4, 15.), vec3<f32>(3., 2.4, 15.), vec3<f32>(3., 2.4, 11.25)), vec2<f32>(0.), vec2<f32>(1.), 7u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(3., 2.4, 11.25), vec3<f32>(1., 2.4, 11.25), vec3<f32>(1., 2.4, 15.)), vec2<f32>(0.), vec2<f32>(1.), 7u),

    // light 3
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-1.25, 2.4, 12.), vec3<f32>(1.25, 2.4, 12.), vec3<f32>(1.25, 2.4, 8.25)), vec2<f32>(0.), vec2<f32>(1.), 8u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(1.25, 2.4, 8.25), vec3<f32>(-1.25, 2.4, 8.25), vec3<f32>(-1.25, 2.4, 12.)), vec2<f32>(0.), vec2<f32>(1.), 8u),
    */

    // light 2
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-1.25, 2.4, 15.), vec3<f32>(1.25, 2.4, 15.), vec3<f32>(1.25, 2.4, 11.25)), vec2<f32>(0.), vec2<f32>(1.), 0u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(1.25, 2.4, 11.25), vec3<f32>(-1.25, 2.4, 11.25), vec3<f32>(-1.25, 2.4, 15.)), vec2<f32>(0.), vec2<f32>(1.), 0u),
);

const NUM_EMISSIVE: u32 = 2u;
var<private> emissive_triangles: array<u32, NUM_EMISSIVE> = array<u32, NUM_EMISSIVE>(12u, 13u/*, 14u, 15u, 16u, 17u*/);

fn intersect_stuff(ray: Ray, out_intersection: ptr<function, Intersection>) -> bool {
    var intersection: Intersection = *out_intersection;

    let tri = Triangle(
        array<vec3<f32>, 3>(
            vec3<f32>(2., -1., 14.),
            vec3<f32>(2.5, -1., 14.),
            vec3<f32>(2.5, -0.5, 14.),
        ),
        vec2<f32>(0.), vec2<f32>(1.),
        4u,
    );

    var intersected = false;

    for (var i = 0u; i < NUM_SPHERES; i++) {
        var cur: Intersection;
        if sphere_intersect(spheres[i], ray, intersection.distance, &cur) {
            intersected = true;
            intersection = pick_intersection(intersection, cur);
        }
    }

    for (var i = 0u; i < NUM_TRIANGLES; i++) {
        var cur: Intersection;
        if triangle_intersect(triangles[i], ray, intersection.distance, &cur) {
            intersected = true;
            intersection = pick_intersection(intersection, cur);
        }
    }

    //intersected = intersected | triangle_intersect(tri, ray, intersection.distance, &intersection);

    *out_intersection = intersection;

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
        sum += abs(dot(intersection.normal, vector_towards_light)) * p * (1. / sample.pdf);
    }

    return sum;
}

fn get_wi_and_weight(intersection: Intersection, was_specular: ptr<function, bool>) -> vec4<f32> {
    let material = materials[intersection.material_idx];
    let going_in = dot(intersection.wo, intersection.normal) > 0.;
    let oriented_normal = select(-intersection.normal, intersection.normal, going_in);

    var wi: vec3<f32>;
    var cos_brdf_over_wi_pdf: f32;

    if material.mat_type == 0u {
        *was_specular = false;
        /*var sample_probability: f32;
        let sphere_sample = sample_sphere_3d(&sample_probability);
        wi = select(sphere_sample, -sphere_sample, dot(sphere_sample, oriented_normal) < 0.);
        sample_probability *= 2.;*/

        var sample_probability: f32;
        let importance_sample = sample_cos_hemisphere_3d(&sample_probability);
        wi = intersection.refl_to_surface * importance_sample;

        let brdf = FRAC_1_PI * 0.5;
        cos_brdf_over_wi_pdf = dot(wi, oriented_normal) * brdf / sample_probability;
    } else if material.mat_type == 1u {
        *was_specular = true;
        wi = reflect(intersection.wo, oriented_normal);
        cos_brdf_over_wi_pdf = 1.;
    } else if material.mat_type == 2u {
        *was_specular = true;
        cos_brdf_over_wi_pdf = 1.;

        let transmittant_index = select(1., material.index, going_in);
        let incident_index     = select(material.index, 1., going_in);

        var refraction: vec3<f32>;
        var p_refraction: f32;
        if refract(
            intersection.wo, oriented_normal,
            incident_index, transmittant_index,
            &p_refraction, &refraction
        ) {
            let generated = rand();

            if generated > p_refraction {
                wi = refraction;
            } else {
                wi = reflect(intersection.wo, oriented_normal);
            }
        } else {
            wi = reflect(intersection.wo, oriented_normal);
        }
    }

    return vec4<f32>(wi, cos_brdf_over_wi_pdf);
}

struct PixelSample {
    rt: vec3<f32>,
    albedo: vec3<f32>,
    normal: vec3<f32>,
    position: vec3<f32>,
    depth: f32,
}

fn pixel_sample_add(lhs: PixelSample, rhs: PixelSample) -> PixelSample {
    return PixelSample(
        lhs.rt + rhs.rt,
        lhs.albedo + rhs.albedo,
        lhs.normal + rhs.normal,
        lhs.position + rhs.position,
        lhs.depth + rhs.depth,
    );
}

fn pixel_sample_div(lhs: PixelSample, divisor: f32) -> PixelSample {
    return PixelSample(
        lhs.rt / divisor,
        lhs.albedo / divisor,
        lhs.normal / divisor,
        lhs.position / divisor,
        lhs.depth / divisor,
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

fn actual_cs(pixel: vec2<u32>, dimensions: vec2<u32>) -> PixelSample {
    let camera = PinpointCamera(FRAC_PI_4);

    var sample_info: PixelSample;

    let ray_primary = pinpoint_generate_ray(camera, pixel, dimensions, vec3<f32>(0.));
    var intersection_primary: Intersection = dummy_intersection(ray_primary);
    if !intersect_stuff(ray_primary, &intersection_primary) {
        return sample_info;
    }
    let material_primary = materials[intersection_primary.material_idx];

    sample_info.albedo = material_primary.albedo;
    sample_info.normal = intersection_primary.normal;
    sample_info.position = intersection_primary.position;
    sample_info.depth = intersection_primary.distance;

    if material_primary.mat_type == 1u {
        let oriented_normal = intersection_oriented_normal(intersection_primary);
        let wi = reflect(intersection_primary.wo, oriented_normal);
        let offset = oriented_normal * 0.0009;
        let secondary_ray = Ray(intersection_primary.position + offset, wi);
        var secondary_intersection = dummy_intersection(ray_primary);
        if intersect_stuff(secondary_ray, &secondary_intersection) {
            sample_info.normal = secondary_intersection.normal;
            sample_info.position = secondary_intersection.position;
            sample_info.depth = secondary_intersection.distance;
        }
    }

    var radiance: vec3<f32>;
    let samples = 16u;
    for (var i = 0u; i < samples; i++) {
        radiance += radiance(intersection_primary);
    }
    sample_info.rt = radiance / f32(samples) + material_primary.emittance;

    return sample_info;

}

fn normal_at(pos: vec2<i32>) -> vec3<f32> { return geometry_buffer[gb_idx_i(pos)].normal_and_depth.xyz; }
fn depth_at(pos: vec2<i32>) -> f32 { return geometry_buffer[gb_idx_i(pos)].normal_and_depth.w; }
fn albedo_at(pos: vec2<i32>) -> vec3<f32> { return geometry_buffer[gb_idx_i(pos)].albedo.xyz; }
fn pos_at(pos: vec2<i32>) -> vec3<f32> { return geometry_buffer[gb_idx_i(pos)].position.xyz; }

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
    let center_geo = geometry_buffer[gb_idx_i(tex_coords)];

    var sum = vec3<f32>(0.);
    var kernel_sum = 0.;

    let σ_rt = 0.85;
    let σ_n  = 0.3;
    let σ_p  = 0.5;

    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            //let kernel_weight = kernel[(x + 2) + ((y + 2) * 5)];
            let kernel_weight = kernel[2 - clamp(2 - abs(x), 0, 2 - abs(y))];

            let offset = vec2<i32>(x, y) * step_scale;
            let cur_coords = clamp(tex_coords + offset, vec2<i32>(0), tex_dims);

            let sample_rt = textureLoad(tex_from, cur_coords).xyz;
            let dist_rt = distance(center_rt, sample_rt);
            let weight_rt = min(exp(-dist_rt / (σ_rt * σ_rt)), 1.);

            let sample_normal = normal_at(cur_coords);
            let dist_normal = distance(center_geo.normal_and_depth.xyz, sample_normal);
            let weight_normal = min(exp(-dist_normal / (σ_n * σ_n)), 1.);

            let sample_pos = pos_at(cur_coords);
            let dist_pos = distance(center_geo.position.xyz, sample_pos);
            let weight_pos = min(exp(-dist_pos / (σ_p * σ_p)), 1.);

            /*let sample_depth = depth_at(cur_coords);
            let dist_depth = abs(sample_depth - center_geo.normal_and_depth.w);
            let weight_depth = min(exp(-dist_depth / (σ_p * σ_p)), 1.);*/

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
    let pixel = global_id.xy;
    let texture_dimensions = textureDimensions(texture_rt);

    setup_rng(global_id.xy, texture_dimensions, local_idx);

    let sample = actual_cs(pixel, texture_dimensions);

    textureStore(texture_rt, pixel, vec4<f32>(sample.rt, 1.));
    geometry_buffer[gb_idx_u(pixel)].normal_and_depth = vec4<f32>(sample.normal, sample.depth);
    geometry_buffer[gb_idx_u(pixel)].albedo = vec4<f32>(sample.albedo, 1.);
    geometry_buffer[gb_idx_u(pixel)].position = vec4<f32>(sample.position, 1.);

    storageBarrier();
    textureStore(texture_denoise_0, pixel, vec4<f32>(a_trous(vec2<i32>(pixel), vec2<i32>(texture_dimensions), 1, texture_rt), 1.));

    storageBarrier();
    textureStore(texture_denoise_1, pixel, vec4<f32>(a_trous(vec2<i32>(pixel), vec2<i32>(texture_dimensions), 2, texture_denoise_0), 1.));

    storageBarrier();
    textureStore(texture_denoise_0, pixel, vec4<f32>(a_trous(vec2<i32>(pixel), vec2<i32>(texture_dimensions), 3, texture_denoise_1), 1.));
}
