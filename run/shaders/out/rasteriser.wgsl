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
struct GeometryElement {
    albedo: vec3<f32>,
    variance: f32,
    normal: vec3<f32>,
    depth: f32,
    position: vec3<f32>,
    distance_from_origin: f32,
    object_index: u32,
}

/*
Layout:
[albedo r][albedo g][albedo b][]
[[normal θ]][[normal φ]]
[[variance]][[depth]]
[[[[position X]]]]

[[[[position Y]]]]
[[[[position Z]]]]
[[[object index]]][]
[[[[distance from origin]]]]
*/
struct PackedGeometry {
    pack_0: vec4<u32>,
    pack_1: vec4<u32>,
}

fn normal_to_spherical(normal: vec3<f32>) -> vec2<f32> {
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

fn normal_from_spherical(spherical: vec2<f32>) -> vec3<f32> {
    let r = 1.;
    let θ = spherical.x;
    let φ = spherical.y;

    let x = r * sin(θ) * cos(φ);
    let y = r * sin(θ) * sin(φ);
    let z = r * cos(θ);

    return vec3<f32>(x, y, z);
}

fn pack_geo(elem: GeometryElement) -> PackedGeometry {
    let albedo_pack = pack4x8unorm(vec4<f32>(elem.albedo, 0.));

    let normal_spherical = normal_to_spherical(elem.normal);
    let normal_pack = pack2x16unorm(vec2<f32>(
        normal_spherical.x * FRAC_1_PI,
        (normal_spherical.y + PI) * FRAC_1_PI,
    ));

    let variance_depth_pack = pack2x16unorm(vec2<f32>(
        elem.variance,
        elem.depth,
    ));

    let pos = vec3<u32>(
        bitcast<u32>(elem.position.x),
        bitcast<u32>(elem.position.y),
        bitcast<u32>(elem.position.z),
    );

    let object_index_pack = elem.object_index & 0x00FFFFFF;
    let distance = bitcast<u32>(elem.distance_from_origin);

    return PackedGeometry(
        vec4<u32>(
            albedo_pack,
            normal_pack,
            variance_depth_pack,
            pos.x,
        ), vec4<u32>(
            pos.y,
            pos.z,
            object_index_pack,
            distance,
        )
    );
}

fn unpack_geo(geo: PackedGeometry) -> GeometryElement {
    let variance_depth = unpack2x16unorm(geo.pack_0[2]);
    let spherical_normal = unpack2x16unorm(geo.pack_0[1]);

    return GeometryElement(
        /* albedo   */ unpack4x8unorm(geo.pack_0[0]).xyz,
        /* variance */ variance_depth.x,
        /* normal   */ normal_from_spherical(spherical_normal),
        /* depth    */ variance_depth.y,
        /* position */ vec3<f32>(
            bitcast<f32>(geo.pack_0[3]),
            bitcast<f32>(geo.pack_1[0]),
            bitcast<f32>(geo.pack_1[1]),
        ),
        /* distance */ bitcast<f32>(geo.pack_1[3]),
        /* index    */ geo.pack_1[2] & 0x00FFFFFF,
    );
}

fn collect_geo_i(coords: vec2<i32>) -> GeometryElement {
    return collect_geo_u(vec2<u32>(max(coords, vec2<i32>(0))));
}
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
@group(1) @binding(0) var previous_frame_pt: texture_2d<f32>;
@group(1) @binding(1) var integrated_frame_pt: texture_2d<f32>;

fn collect_geo_u(coords: vec2<u32>) -> GeometryElement {
    var ret: GeometryElement;
    return ret;
}

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

    let prev = textureLoad(previous_frame_pt, pixel, 0);
    let integrated = textureLoad(integrated_frame_pt, pixel, 0);

    return FragmentOutput(
        vec4<f32>(in.color, abs(dot(prev, prev) - dot(integrated, integrated))),
        vec4<f32>(in.normal, in.position.z),
        vec4<f32>(in.scene_position, length(in.scene_position)),
        in.triangle_index,
    );
}
