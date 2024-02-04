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

var<private> wg_rng: RNGState;
var<private> rng_mix_value: u32;
var<private> rng_permuter: bool = false;

fn rand() -> f32 {
    workgroupBarrier();
    let base_res = rng_scramble_impl_xoroshiro128pp(wg_rng.state);
    workgroupBarrier();

    // kinda racy
    if rng_permuter {
        wg_rng.state = rng_permute_impl_xoroshiro128(wg_rng.state);
    }

    let u32_val = base_res ^ rng_mix_value;

    return ldexp(f32(u32_val & 0xFFFFFFu), -24);
}

fn setup_rng(pixel: vec2<u32>, dims: vec2<u32>, local_idx: u32) {
    let pix_hash = vec2_u32_hash(pixel);

    // why is this so slow?
    // i mean, obviously, because of some memory bottleneck
    // but why is it as slow as it is???
    let pix_seed = textureLoad(texture_noise, pixel % textureDimensions(texture_noise));

    // if local_idx == 0u {
        wg_rng = RNGState(setup_rng_impl_xoroshiro128(pixel, dims, pix_hash, pix_seed));
        rng_permuter = true;
    // }

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

fn sample_sphere_3d() -> vec3<f32> {
    // avoid messing with the cache for the two
    let norm_vec = vec3<f32>(
        box_muller_map(vec2<f32>(
            rand(),
            rand(),
        )),
        box_muller(),
    );

    return normalize(norm_vec);
}
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Intersection {
    distance: f32,
    position: vec3<f32>,
    normal: vec3<f32>,
    wo: vec3<f32>,
    material_idx: u32,
}

fn dummy_intersection(ray: Ray) -> Intersection {
    let inf = 1. / 0.;
    return Intersection(inf, vec3<f32>(inf), -ray.direction, -ray.direction, 0u);
}

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

fn sphere_intersect(sphere: Sphere, ray: Ray, best: f32, out: ptr<function, Intersection>) -> bool {
    let direction = ray.origin - sphere.center;

    let a = 1.;
    let b = 2. * dot(direction, ray.direction);
    let c = dot(direction, direction) - (sphere.radius * sphere.radius);

    var t = 0.;
    if !solve_sphere_quadratic(a, b, c, &t) {
        return false;
    }

    let isect_pos = ray.origin + ray.direction * t;
    let local_pos = isect_pos - sphere.center;
    let normal = normalize(local_pos);

    *out = Intersection(
        t,
        isect_pos,
        normal,
        -ray.direction,
        sphere.material,
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

struct Material {
    albedo: vec3<f32>,
    emittance: vec3<f32>,
    // 0 -> diffuse/specular, 1 -> dielectric
    mat_type: u32,
    reflectivity: f32,
    index: f32,
}

// PT modelled after https://youtu.be/FewqoJjHR0A?t=1112

var<private> materials: array<Material, 6> = array<Material, 6>(
    Material(vec3<f32>(0.)  , vec3<f32>(12.), 0u, 0.  , 1. ),         // 0 light
    Material(vec3<f32>(0.99), vec3<f32>(0.) , 0u, 1., 1. ),         // 1 mirror
    Material(vec3<f32>(0.99), vec3<f32>(0.) , 1u, 0.  , 1.7),         // 2 glass
    Material(vec3<f32>(0.75), vec3<f32>(0.) , 0u, 0.  , 1. ),         // 3 white
    Material(vec3<f32>(0.75, 0.25, 0.25), vec3<f32>(0.), 0u, 0., 1.), // 4 red
    Material(vec3<f32>(0.25, 0.25, 0.75), vec3<f32>(0.), 0u, 0., 1.), // 5 blue
);

var<private> spheres: array<Sphere, 9> = array<Sphere, 9>(
    Sphere(vec3<f32>(-1.75 , -2.5 + 0.9      , 17.5  ), .9     , 1u), // mirror
    Sphere(vec3<f32>(1.75  , -2.5 + 0.9 + 0.2, 16.5  ), .9     , 2u), // glass
    Sphere(vec3<f32>(0.    , 42.499          , 15.   ), 40.    , 0u), // light
    Sphere(vec3<f32>(0.    , 0.              , -5000.), 4980.  , 3u), // front wall
    Sphere(vec3<f32>(0.    , 0.              , 5000. ), 4980.  , 3u), // backwall
    Sphere(vec3<f32>(5000. , 0.              , 0.    ), 4996.5 , 5u), // right wall
    Sphere(vec3<f32>(0.    , 5000.           , 5.    ), 4997.5 , 3u), // ceiling
    Sphere(vec3<f32>(-5000., 0.              , 0.    ), 4996.5 , 4u), // left wall
    Sphere(vec3<f32>(0.    , -5000.          , 5.    ), 4997.5 , 3u), // floor
);

fn intersect_stuff(ray: Ray, out_intersection: ptr<function, Intersection>) -> bool {
    var intersection: Intersection = *out_intersection;

    var intersected = false;
    for (var i = 0u; i < 9u; i++) {
        var cur: Intersection;
        if sphere_intersect(spheres[i], ray, intersection.distance, &cur) {
            intersected = true;
            intersection = pick_intersection(intersection, cur);
        }
    }

    *out_intersection = intersection;

    return intersected;
}

fn sample_pixel(camera: PinpointCamera, pixel: vec2<u32>, dimensions: vec2<u32>) -> vec3<f32> {
    //var ray = pinpoint_generate_ray(camera, pixel, dimensions, vec3<f32>(0., 0., uniforms.current_instant * 2.));
    var ray = pinpoint_generate_ray(camera, pixel, dimensions, vec3<f32>(0.));

    var attenuation = vec3<f32>(1.);
    var light = vec3<f32>(0.);

    for (var i = 0u; i < 4u; i++) {
        var intersection: Intersection = dummy_intersection(ray);
        if !intersect_stuff(ray, &intersection) {
            break;
        }

        let material = materials[intersection.material_idx];

        let going_in = dot(ray.direction, intersection.normal) < 0.;
        let oriented_normal = select(-intersection.normal, intersection.normal, going_in);

        var wi: vec3<f32>;
        var cos_brdf_over_wi_pdf: f32;
        
        if material.mat_type == 0u {
            let refl_sample = rand();
            if refl_sample <= material.reflectivity {
                wi = reflect(intersection.wo, oriented_normal);
                cos_brdf_over_wi_pdf = material.reflectivity;
            } else {
                let sphere_sample = sample_sphere_3d();
                wi = select(sphere_sample, -sphere_sample, dot(sphere_sample, intersection.normal) < 0.);
                let wi_pdf = FRAC_1_PI * 0.5 * (1. - material.reflectivity);
                let brdf = FRAC_1_PI * 0.5;
                cos_brdf_over_wi_pdf = dot(wi, intersection.normal) * brdf / wi_pdf;
            }
        } else if material.mat_type == 1u {
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

        ray = Ray(intersection.position + intersection.normal * 0.0009 * select(1., -1., dot(wi, intersection.normal) < 0.), wi);
        light = light + material.emittance * attenuation;

        let cur_attenuation = material.albedo * cos_brdf_over_wi_pdf;
        attenuation = cur_attenuation * attenuation;
    }

    return light;
}

fn actual_cs(pixel: vec2<u32>, dimensions: vec2<u32>, previous_value: vec4<f32>) -> vec4<f32> {
    let camera = PinpointCamera(FRAC_PI_4);

    let samples = 6u;
    var res = vec3<f32>(0.);
    for (var i = 0u; i < samples; i++) {
        res += sample_pixel(camera, pixel, dimensions);
    }
    res /= f32(samples);

    let w_prev = previous_value * f32(uniforms.frame_no) / f32(uniforms.frame_no + 1u);
    let w_new  = vec4<f32>(res / f32(uniforms.frame_no + 1u), 1.);

    return w_prev + w_new;
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

    setup_rng(workgroup_id.xy, texture_dimensions, local_idx);
    // workgroupBarrier();

    /*rng.state[0] = rng_next(&rng);
    rng.state[1] = rng_next(&rng);
    rng.state[2] = rng_next(&rng);
    rng.state[3] = rng_next(&rng);*/

    var previous_value: vec4<f32>;
    if texture_selection == 0u {
        previous_value = textureLoad(texture_1, vec2<i32>(pixel));
    } else {
        previous_value = textureLoad(texture_0, vec2<i32>(pixel));
    }

    let out_color = actual_cs(pixel, texture_dimensions, previous_value);

    if texture_selection == 0u {
        textureStore(texture_0, vec2<i32>(pixel), out_color);
    } else {
        textureStore(texture_1, vec2<i32>(pixel), out_color);
    }
}
