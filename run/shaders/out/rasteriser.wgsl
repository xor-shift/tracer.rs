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
struct State {
    camera_transform: mat4x4<f32>,
    frame_seed: vec4<u32>,
    camera_position: vec3<f32>,
    frame_no: u32,
    current_instant: f32,
    width: u32,
    height: u32,
    visualisation_mode: i32,
}
struct RasteriserUniform {
    camera: mat4x4<f32>,
    width: u32,
    height: u32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) material: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) scene_position: vec3<f32>,
    @location(3) triangle_index: u32,
}

@group(0) @binding(0) var<uniform> uniforms: State;
//@group(1) @binding(0) var<storage, read_write> geometry_buffer: array<GeometryElement>;

var<private> MATERIAL_COLORS: array<vec3<f32>, 9> = array<vec3<f32>, 9>(
    vec3<f32>(1., 1., 1.),
    vec3<f32>(0.99, 0.99, 0.99),
    vec3<f32>(0.99, 0.99, 0.99),
    vec3<f32>(0.75, 0.75, 0.75),
    vec3<f32>(0.75, 0.25, 0.25),
    vec3<f32>(0.25, 0.25, 0.75),
    vec3<f32>(1., 0., 0.),
    vec3<f32>(0., 1., 0.),
    vec3<f32>(0., 0., 1.),
);

@vertex fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    vert: VertexInput,
) -> VertexOutput {
    return VertexOutput(
        uniforms.camera_transform * vec4<f32>(vert.position, 1.),
        MATERIAL_COLORS[vert.material],
        vert.normal,
        vert.position,
        vertex_index / 3u,
    );
}

struct FragmentOutput {
    @location(0) albdeo: vec4<f32>,
    @location(1) pack_normal_depth: vec4<f32>,
    @location(2) pack_positon_distance: vec4<f32>,
    @location(3) object_index: u32,
}

@fragment fn fs_main(
    in: VertexOutput,
) -> FragmentOutput {
    let test = in.position.xyz / in.position.w;
    let pixel = vec2<u32>(trunc(in.position.xy));

    return FragmentOutput(
        vec4<f32>(in.color, 1.),
        vec4<f32>(in.normal, in.position.z),
        vec4<f32>(in.scene_position, length(in.scene_position)),
        in.triangle_index,
    );
}
