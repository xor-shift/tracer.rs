struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    direction_reciprocal: vec3<f32>,
}

fn ray_new(origin: vec3<f32>, direction: vec3<f32>) -> Ray {
    return Ray(origin, direction, 1. / direction);
}

fn ray_from_pixel(pixel: vec2<u32>, state: State) -> Ray {
    let dimensions = vec2<f32>(state.dimensions);
    let pixel_corr = vec2<f32>(f32(pixel.x), dimensions.y - f32(pixel.y));
    
    // the 1.5 fixes the fov issue and i have no clue why
    let ndc_pixel = ((pixel_corr / dimensions) * 2. - 1.) * 1.5;

    let coord = state.inverse_transform * vec4<f32>(ndc_pixel, 0., 1.);
    let ray_dir = normalize((coord.xyz / coord.w) - state.camera_position);

    return ray_new(state.camera_position, ray_dir);
}

struct Material {
    mat_type: u32, // determines how the data is interpreted (min 0, max 255)
    /*
        all values are in the range [0, 1) but fields may have scale factors

        | symbol | name      | scale | resolution (bits) | type  |
        +--------+-----------+-------+-------------------+-------+
        | *      | unused    | N/A   | N/A               | N/A   |
        | R      | red       | 1     | 8                 | unorm |
        | G      | green     | 1     | 8                 | unorm |
        | B      | blue      | 1     | 8                 | unorm |
        | g      | gloss     | NYD   | 16                | unorm |
        | I      | r. idx    | NYD   | 16                | unorm |
        | i      | intensity | 100   | 16                | unorm |

        materials:
        0 -> **** air
        1 -> RGB* diffuse
        2 -> RGBi light
        3 -> RGB* perfect mirror
        4 -> RGBI glass
        5 -> RGBG glossy

        brute force thing for white lights of arbitrary intensitites:
        function f(target) {
            for (let unorm8 = 0; unorm8 < 256; unorm8++) {
                let unorm8_f32 = unorm8 / 255;

                let optim_scaled = target / unorm8_f32;
                let optim_unorm = optim_scaled / 100;
                let optim_unorm16 = optim_unorm * 65535;

                //
            }
        }
    */
    data: vec4<f32>,
}

// packs a material for storage
fn material_pack(material: Material) -> vec2<u32> {
    let first_quad = (pack4x8unorm(material.data) & 0x00FFFFFFu) | ((material.mat_type & 0x000000FFu) << 24u);
    let second_quad = pack2x16unorm(material.data.ba) & 0xFFFF0000u;

    return vec2<u32>(first_quad, second_quad);
}

// unpacks a packed material
fn material_unpack(pack: vec2<u32>) -> Material {
    let mat_type = (pack[0] >> 24u) & 0xFFu;
    let mat_data_rgb = unpack4x8unorm(pack[0]).rgb;
    let mat_data_a = unpack2x16unorm(pack[1]).x;

    return Material(
        /* mat_type */ mat_type,
        /* data     */ vec4<f32>(mat_data_rgb, mat_data_a),
    );
}

struct Intersection {
    material: Material,

    wo: vec3<f32>,
    t: f32,

    gloabl_pt: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
}

fn intersecton_new_dummy() -> Intersection {
    return Intersection(
        /* material  */ Material (
            /* typ */ 0,
            /* dat */ vec4<f32>(0.),
        ),

        /* wo        */ vec3<f32>(0.),
        /* t         */ 99999999.,

        /* global_pt */ vec3<f32>(0.),
        /* normal    */ vec3<f32>(0.),
        /* uv        */ vec2<f32>(0.),
    );
}
