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
