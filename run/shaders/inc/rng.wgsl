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

//var<workgroup> wg_rng: RNGState;
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
