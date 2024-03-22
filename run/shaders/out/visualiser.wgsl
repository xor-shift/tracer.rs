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
struct Geometry {
    normal: vec3<f32>,
    position: vec3<f32>,
}

fn _geometry_normal_to_spherical(normal: vec3<f32>) -> vec2<f32> {
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

fn _geometry_normal_from_spherical(spherical: vec2<f32>) -> vec3<f32> {
    let r = 1.;
    let θ = spherical.x;
    let φ = spherical.y;

    let x = r * sin(θ) * cos(φ);
    let y = r * sin(θ) * sin(φ);
    let z = r * cos(θ);

    return vec3<f32>(x, y, z);
}

fn geometry_pack(geometry: Geometry) -> PackedGeometry {
    return PackedGeometry(
        /* pack_0 */ vec4<u32>(0),
        /* pack_1 */ vec4<u32>(0),
    );
}

struct PackedGeometry {
    pack_0: vec4<u32>,
    pack_1: vec4<u32>,
}
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
fn material_from_pack(pack: vec2<u32>) -> Material {
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
struct State {
    camera_transform: mat4x4<f32>,
    inverse_transform: mat4x4<f32>,
    frame_seed: vec4<u32>,
    camera_position: vec3<f32>,
    frame_no: u32,
    dimensions: vec2<u32>,
    current_instant: f32,
    visualisation_mode: i32,
}
var<private> VISUALISER_VERTICES: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-1., 1.,),
    vec2<f32>(-1., -1.,),
    vec2<f32>(1., -1.,),
    vec2<f32>(1., 1.,),
);

var<private> VISUALISER_INDICES: array<u32, 6> = array<u32, 6>(0, 1, 2, 2, 3, 0);

var<private> VISUALISER_UVS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0., 0.),
    vec2<f32>(0., 1.),
    vec2<f32>(1., 1.),
    vec2<f32>(1., 0.),
);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@group(0) @binding(0) var<uniform> state: State;
@group(1) @binding(0) var texture_rt: texture_2d<f32>;
@group(1) @binding(1) var texture_geo: texture_2d_array<u32>;
@group(1) @binding(2) var texture_denoise: texture_2d_array<f32>;

@vertex fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = VISUALISER_UVS[VISUALISER_INDICES[vertex_index]];
    out.position = vec4<f32>(VISUALISER_VERTICES[VISUALISER_INDICES[vertex_index]], 0., 1.0);
    return out;
}

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_pos = vec2<u32>(vec2<f32>(state.dimensions) * in.tex_coords);

    let geo_0 = textureLoad(texture_geo, tex_pos, 0, 0);
    let geo_1 = textureLoad(texture_geo, tex_pos, 1, 0);

    let denoise_0 = textureLoad(texture_denoise, tex_pos, 0, 0);
    let denoise_1 = textureLoad(texture_denoise, tex_pos, 1, 0);

    switch state.visualisation_mode {
        //default: { return vec4<f32>(1., 0., 0., 0.); }
        default: { return textureLoad(texture_rt, tex_pos, 0); }
    }
}