struct MainUniform {
    // at frame no 0, texture 1 should be used and texture 0 should be drawn on
    frame_no: u32,
    current_instant: f32,
};

@group(0) @binding(0) var<uniform> uniforms: MainUniform;
@group(1) @binding(0) var texture_0: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(1) var texture_1: texture_storage_2d<rgba8unorm, read_write>;

fn rotl(v: u32, amt: u32) -> u32 {
    return (v << amt) | (v >> (32u - amt));
}

struct XoroshiroState {
    state: array<u32, 4>,
}

fn xoroshiro_next(state: ptr<private, XoroshiroState>) -> u32 {
    let sp = &((*state).state);
    let ret = rotl((*sp)[0] + (*sp)[3], 7u) + (*sp)[0];

    let t = (*sp)[1] << 9u;

    (*sp)[2] ^= (*sp)[0];
    (*sp)[3] ^= (*sp)[1];
    (*sp)[1] ^= (*sp)[2];
    (*sp)[0] ^= (*sp)[3];

    (*sp)[2] ^= t;
    (*sp)[3] = rotl((*sp)[3], 11u);

    return ret;
}

fn xoroshiro_next_unorm(state: ptr<private, XoroshiroState>) -> f32 {
    let res_u32 = xoroshiro_next(state) & 0xFFFFFFu;
    let res_f32 = f32(res_u32);
    let res_unorm = ldexp(res_f32, -24);

    return res_unorm;
}

const XOROSHIRO_SHORT_JUMP: array<u32, 4> = array<u32, 4>(0x8764000bu, 0xf542d2d3u, 0x6fa035c3u, 0x77f2db5bu);
const XOROSHIRO_LONG_JUMP: array<u32, 4> = array<u32, 4>(0xb523952eu, 0x0b6f099fu, 0xccf5a0efu, 0x1c580662u);

fn xoroshiro_jump(state: ptr<private, XoroshiroState>, long_jump: bool) {
    let sp = &((*state).state);

    var s = array<u32, 4>(0u, 0u, 0u, 0u);

    for (var b = 0u; b < 32u; b++) {
        if (select(XOROSHIRO_SHORT_JUMP[0], XOROSHIRO_LONG_JUMP[0], long_jump) & (1u << b)) != 0u {
            for (var i = 0u; i < 4u; i++) {
                s[i] ^= (*sp)[i];
            }

            xoroshiro_next(state);
        }
    }

    for (var b = 0u; b < 32u; b++) {
        if (select(XOROSHIRO_SHORT_JUMP[1], XOROSHIRO_LONG_JUMP[1], long_jump) & (1u << b)) != 0u {
            for (var i = 0u; i < 4u; i++) {
                s[i] ^= (*sp)[i];
            }

            xoroshiro_next(state);
        }
    }

    for (var b = 0u; b < 32u; b++) {
        if (select(XOROSHIRO_SHORT_JUMP[2], XOROSHIRO_LONG_JUMP[2], long_jump) & (1u << b)) != 0u {
            for (var i = 0u; i < 4u; i++) {
                s[i] ^= (*sp)[i];
            }

            xoroshiro_next(state);
        }
    }

    for (var b = 0u; b < 32u; b++) {
        if (select(XOROSHIRO_SHORT_JUMP[3], XOROSHIRO_LONG_JUMP[3], long_jump) & (1u << b)) != 0u {
            for (var i = 0u; i < 4u; i++) {
                s[i] ^= (*sp)[i];
            }

            xoroshiro_next(state);
        }
    }

    for (var i = 0u; i < 4u; i++) {
        (*sp)[i] = s[i];
    }
}

var<private> xoroshiro_state: XoroshiroState = XoroshiroState(array<u32, 4>(0u, 0u, 0u, 0u));

fn actual_cs(pixel: vec2<u32>, dimensions: vec2<u32>) -> vec4<f32> {
    xoroshiro_state.state = array<u32, 4>(pixel.x, dimensions.x, pixel.y, dimensions.y);
    xoroshiro_jump(&xoroshiro_state, false);
    xoroshiro_jump(&xoroshiro_state, true);
    xoroshiro_jump(&xoroshiro_state, false);
    xoroshiro_jump(&xoroshiro_state, true);

    let rng_res = xoroshiro_next_unorm(&xoroshiro_state);
    return vec4<f32>(rng_res, rng_res, rng_res, 1.0);

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
 
@compute @workgroup_size(1) fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let texture_selection = select(0u, 1u, uniforms.frame_no % 2u == 0u);
    let texture_size = select(textureDimensions(texture_0), textureDimensions(texture_1), texture_selection == 0u);

    let out_color = actual_cs(global_id.xy, texture_size);

    if texture_selection == 0u {
        textureStore(texture_0, vec2<i32>(global_id.xy), out_color);
    } else {
        textureStore(texture_1, vec2<i32>(global_id.xy), out_color);
    }
}
