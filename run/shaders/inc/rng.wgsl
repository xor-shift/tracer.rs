fn rotl(v: u32, amt: u32) -> u32 {
    return (v << amt) | (v >> (32u - amt));
}

struct RNGState {
    state: array<u32, 2>,
}

fn rng_next_impl_xoroshiro128pp(s: ptr<function, array<u32, 4>>) -> u32 {
    let ret = rotl((*s)[0] + (*s)[3], 7u) + (*s)[0];

    let t = (*s)[1] << 9u;

    (*s)[2] ^= (*s)[0];
    (*s)[3] ^= (*s)[1];
    (*s)[1] ^= (*s)[2];
    (*s)[0] ^= (*s)[3];

    (*s)[2] ^= t;
    (*s)[3] = rotl((*s)[3], 11u);

    return ret;
}

fn rng_next_impl_xoroshiro64s(s: ptr<function, array<u32, 2>>) -> u32 {
    let s0 = (*s)[0];
	let result = s0 * 0x9E3779BBu;

	let s1 = (*s)[1] ^ s0;
	(*s)[0] = rotl(s0, 26u) ^ s1 ^ (s1 << 9u);
	(*s)[1] = rotl(s1, 13u);

	return result;
}

fn rng_next(state: ptr<function, RNGState>) -> u32 {
    // https://github.com/gfx-rs/wgpu/issues/4549
    //return rng_next_impl_xoroshiro128(&((*state).state));

    var arr = (*state).state;
    let ret = rng_next_impl_xoroshiro64s(&arr);
    (*state).state = arr;
    return ret;
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
    //let pix_seed = textureLoad(texture_noise, pixel % textureDimensions(texture_noise));
    let pix_seed = vec4<u32>(1234u, 5678u, 2357u, 1337u);

    return RNGState(setup_rng_impl_xoroshiro64(pixel, dims, pix_hash, pix_seed));
}
