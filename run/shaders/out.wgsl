const PI: f32 = 3.14159265358979323846264338327950288; // π
const FRAC_PI_2: f32 = 1.57079632679489661923132169163975144; // π/2
const FRAC_PI_3: f32 = 1.04719755119659774615421446109316763; // π/3
const FRAC_PI_4: f32 = 0.785398163397448309615660845819875721; // π/4
const FRAC_PI_6: f32 = 0.39269908169872415480783042290993786; // π/6
const FRAC_1_PI: f32 = 0.318309886183790671537767526745028724; // 1/π
const FRAC_1_SQRT_PI: f32 = 0.564189583547756286948079451560772586; // 1/sqrt(π)
const FRAC_2_PI: f32 = 0.636619772367581343075535053490057448; // 2/π
const FRAC_2_SQRT_PI: f32 = 1.12837916709551257389615890312154517; // 2/sqrt(π)
const PHI: f32 = 1.618033988749894848204586834365638118; // φ
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

fn rng_next(state: ptr<function, RNGState>) -> u32 {
    // https://github.com/gfx-rs/wgpu/issues/4549
    //return rng_next_impl_xoroshiro128(&((*state).state));

    let ret = rng_scramble_impl_xoroshiro128pp((*state).state);
    (*state).state = rng_permute_impl_xoroshiro128((*state).state);
    return ret;
}

var<workgroup> wg_rng: RNGState;

fn wg_rng_next(local_idx: u32) -> u32 {
    workgroupBarrier();
    let base_res = rng_scramble_impl_xoroshiro128pp(wg_rng.state);
    workgroupBarrier();

    // kinda racy
    if local_idx == 0u {
        wg_rng.state = rng_permute_impl_xoroshiro128(wg_rng.state);
    }

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

    //return base_res ^ wg_schedule[local_idx];
    return base_res * wg_schedule[local_idx];
}

// for xoroshiro128pp
/*const XOROSHIRO_SHORT_JUMP: array<u32, 4> = array<u32, 4>(0x8764000bu, 0xf542d2d3u, 0x6fa035c3u, 0x77f2db5bu);
const XOROSHIRO_LONG_JUMP: array<u32, 4> = array<u32, 4>(0xb523952eu, 0x0b6f099fu, 0xccf5a0efu, 0x1c580662u);

// too slow
fn rng_jump(state: ptr<function, RNGState>, long_jump: bool) {
    let sp = &((*state).state);

    var s = array<u32, 4>(0u, 0u, 0u, 0u);

    for (var b = 0u; b < 32u; b++) {
        if (select(XOROSHIRO_SHORT_JUMP[0], XOROSHIRO_LONG_JUMP[0], long_jump) & (1u << b)) != 0u {
            for (var i = 0u; i < 4u; i++) {
                s[i] ^= (*sp)[i];
            }

            rng_next(state);
        }
    }

    for (var b = 0u; b < 32u; b++) {
        if (select(XOROSHIRO_SHORT_JUMP[1], XOROSHIRO_LONG_JUMP[1], long_jump) & (1u << b)) != 0u {
            for (var i = 0u; i < 4u; i++) {
                s[i] ^= (*sp)[i];
            }

            rng_next(state);
        }
    }

    for (var b = 0u; b < 32u; b++) {
        if (select(XOROSHIRO_SHORT_JUMP[2], XOROSHIRO_LONG_JUMP[2], long_jump) & (1u << b)) != 0u {
            for (var i = 0u; i < 4u; i++) {
                s[i] ^= (*sp)[i];
            }

            rng_next(state);
        }
    }

    for (var b = 0u; b < 32u; b++) {
        if (select(XOROSHIRO_SHORT_JUMP[3], XOROSHIRO_LONG_JUMP[3], long_jump) & (1u << b)) != 0u {
            for (var i = 0u; i < 4u; i++) {
                s[i] ^= (*sp)[i];
            }

            rng_next(state);
        }
    }

    for (var i = 0u; i < 4u; i++) {
        (*sp)[i] = s[i];
    }
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

fn setup_rng(pixel: vec2<u32>, dims: vec2<u32>) -> RNGState {
    let pix_hash = vec2_u32_hash(pixel);

    // why is this so slow?
    // i mean, obviously, because of some memory bottleneck
    // but why is it as slow as it is???
    let pix_seed = textureLoad(texture_noise, pixel % textureDimensions(texture_noise));

    return RNGState(setup_rng_impl_xoroshiro128(pixel, dims, pix_hash, pix_seed));
}
var<private> box_muller_cache: f32;
var<private> box_muller_cache_full: bool = false;

fn box_muller(unorm_samples: vec2<f32>) -> vec2<f32> {
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
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Intersection {
    distance: f32,
    pos: vec3<f32>,
    normal: vec3<f32>,
    wo: vec3<f32>,
}
struct PinpointCamera {
    fov: f32,
}

fn pinpoint_generate_ray(
    camera: PinpointCamera,
    screen_coords: vec2<u32>,
    screen_dims: vec2<u32>,
    pos: vec3<f32>,
    local_idx: u32,
) -> Ray {
    let half_theta = camera.fov / 2.;
    let d = (1. / (2. * sin(half_theta))) * sqrt(abs((f32(screen_dims.x) * (2. - f32(screen_dims.x)))));

    let offset = vec2<f32>(u32_to_f32_unorm(wg_rng_next(local_idx)), u32_to_f32_unorm(wg_rng_next(local_idx)));

    let direction = vec3<f32>(
        f32(screen_coords.x) - (f32(screen_dims.x) / 2.) + offset.x,
        f32(screen_dims.y - screen_coords.y - 1u) - f32(screen_dims.y) / 2. + offset.y,
        d,
    );

    return Ray(pos, direction);
}
struct Sphere {
    center: vec3<f32>,
    radius: f32,
}

// value written to out is indeterminate if the function returns false.
fn solve_sphere_quadratic(a: f32, b: f32, c: f32, out: ptr<function, f32>) -> bool {
    let delta = b * b - 4. * a * c;
    let sqrt_delta = sqrt(delta);

    let raw_sol_0 = (-b + sqrt_delta) / (2. * a);
    let raw_sol_1 = (-b - sqrt_delta) / (2. * a);

    let sol_0 = select(raw_sol_0, 1. / 0., raw_sol_0 >= 0.);
    let sol_1 = select(raw_sol_1, 1. / 0., raw_sol_1 >= 0.);

    let solution = select(sol_0, sol_1, sol_0 < sol_1);

    *out = solution;

    return delta >= 0.;
}

fn sphere_intersect(sphere: Sphere, ray: Ray, out: ptr<function, Intersection>) -> bool {
    let direction = ray.origin - sphere.center;

    let a = 1.;
    let b = 2. * dot(direction, ray.direction);
    let c = dot(direction, direction) - (sphere.radius * sphere.radius);

    var t = 0.;
    if !solve_sphere_quadratic(a, b, c, &t) {
        return false;
    }

    *out = Intersection(
        t,
        ray.origin + ray.direction * t,
        vec3<f32>(0.),
        -ray.direction,
    );

    return true;
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
@group(1) @binding(2) var texture_noise: texture_storage_2d<rgba32uint, read>;

fn actual_cs(pixel: vec2<u32>, dimensions: vec2<u32>, local_idx: u32) -> vec4<f32> {
    let camera = PinpointCamera(FRAC_PI_4);
    let ray = pinpoint_generate_ray(camera, pixel, dimensions, vec3<f32>(0.), local_idx);

    let rng_res = vec3<f32>(u32_to_f32_unorm(wg_rng_next(local_idx)));
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
 
@compute @workgroup_size(8, 8) fn cs_main(
    @builtin(global_invocation_id)   global_id: vec3<u32>,
    @builtin(workgroup_id)           workgroup_id: vec3<u32>,
    @builtin(local_invocation_id)    local_id:  vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
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

    if local_idx == 0u {
        wg_rng = setup_rng(workgroup_id.xy, texture_dimensions);
    }
    workgroupBarrier();

    /*rng.state[0] = rng_next(&rng);
    rng.state[1] = rng_next(&rng);
    rng.state[2] = rng_next(&rng);
    rng.state[3] = rng_next(&rng);*/

    let out_color = actual_cs(pixel, texture_dimensions, local_idx);

    if texture_selection == 0u {
        textureStore(texture_0, vec2<i32>(pixel), out_color);
    } else {
        textureStore(texture_1, vec2<i32>(pixel), out_color);
    }
}
