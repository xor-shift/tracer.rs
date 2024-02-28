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
struct GeometryElement {
    albedo: vec3<f32>,
    variance: f32,
    normal: vec3<f32>,
    depth: f32,
    position: vec3<f32>,
    distance_from_origin: f32,
    object_index: u32,
    was_invalidated: bool,
    similarity_score: f32,
}

/*
Layout:
[albedo r][albedo g][albedo b][]
[[normal θ]][[normal φ]]
[[variance]][[depth]]
[[[[position X]]]]

[[[[position Y]]]]
[[[[position Z]]]]
[bitflags of no specific purpose][[[object index]]]
[[[[]]]]
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
        (normal_spherical.y + PI) * FRAC_1_PI * 0.5,
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

    let object_index_pack = (elem.object_index & 0x00FFFFFFu) | select(0u, 0x80000000u, elem.was_invalidated);
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
            bitcast<u32>(elem.similarity_score),
        )
    );
}

fn unpack_geo(geo: PackedGeometry) -> GeometryElement {
    let variance_depth = unpack2x16unorm(geo.pack_0[2]);
    let spherical_normal = unpack2x16unorm(geo.pack_0[1]);
    let position = vec3<f32>(
        bitcast<f32>(geo.pack_0[3]),
        bitcast<f32>(geo.pack_1[0]),
        bitcast<f32>(geo.pack_1[1]),
    );

    return GeometryElement(
        /* albedo   */ unpack4x8unorm(geo.pack_0[0]).xyz,
        /* variance */ variance_depth.x,
        /* normal   */ normal_from_spherical(vec2<f32>(
            spherical_normal.x * PI,
            (spherical_normal.y * 2. - 1.) * PI,
        )),
        /* depth    */ variance_depth.y,
        /* position */ position,
        /* distance */ length(position),
        /* index    */ geo.pack_1[2] & 0x00FFFFFF,
        /* inval'd  */ (geo.pack_1[2] & 0x80000000u) == 0x80000000u,
        /* s-lity   */ bitcast<f32>(geo.pack_1[3]),
    );
}

fn collect_geo_i(coords: vec2<i32>) -> GeometryElement {
    return collect_geo_u(vec2<u32>(max(coords, vec2<i32>(0))));
}

fn collect_geo_t2d(coords: vec2<u32>, pack_0: texture_2d<u32>, pack_1: texture_2d<u32>) -> GeometryElement {
    let sample_pack_0 = textureLoad(pack_0, coords, 0);
    let sample_pack_1 = textureLoad(pack_1, coords, 0);

    return unpack_geo(PackedGeometry(sample_pack_0, sample_pack_1));
}

fn collect_geo_ts2d(coords: vec2<u32>, pack_0: texture_storage_2d<rgba32uint, read_write>, pack_1: texture_storage_2d<rgba32uint, read_write>) -> GeometryElement {
    let sample_pack_0 = textureLoad(pack_0, coords);
    let sample_pack_1 = textureLoad(pack_1, coords);

    return unpack_geo(PackedGeometry(sample_pack_0, sample_pack_1));
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
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
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

@vertex fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = VISUALISER_UVS[VISUALISER_INDICES[vertex_index]];
    out.position = vec4<f32>(VISUALISER_VERTICES[VISUALISER_INDICES[vertex_index]], 0., 1.0);
    return out;
}

struct MainUniform {
    width: u32,                 // 00..03
    height: u32,                // 04..07
    visualisation_mode: i32,    // 08..0B
}

@group(0) @binding(0) var<uniform> uniforms: MainUniform;
@group(1) @binding(0) var texture_rt: texture_2d<f32>;
@group(1) @binding(1) var texture_geo_pack_0: texture_2d<u32>;
@group(1) @binding(2) var texture_geo_pack_1: texture_2d<u32>;
@group(1) @binding(3) var texture_geo_pack_0_old: texture_2d<u32>;
@group(1) @binding(4) var texture_geo_pack_1_old: texture_2d<u32>;
@group(1) @binding(5) var texture_denoise_0: texture_2d<f32>;
@group(1) @binding(6) var texture_denoise_1: texture_2d<f32>;

fn collect_geo_u(coords: vec2<u32>) -> GeometryElement { return collect_geo_t2d(coords, texture_geo_pack_0, texture_geo_pack_1); }

fn aces_film(x: vec3<f32>) -> vec3<f32> {
    let raw = (x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14);
    return clamp(raw, vec3<f32>(0.), vec3<f32>(1.));
}

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = textureDimensions(texture_rt);
    let tex_pos = vec2<u32>(vec2<f32>(tex_size) * in.tex_coords);

    //return vec4<f32>(vec2<f32>(tex_pos) / vec2<f32>(tex_size), 0., 1.);
    //return vec4<f32>(vec2<f32>(tex_pos) / vec2<f32>(f32(uniforms.width), f32(uniforms.height)), 0., 1.);
    //return vec4<f32>(ge_normal(geometry_buffer[gb_idx_u(tex_pos)]), 1.);
    //return vec4<f32>(geometry_buffer[gb_idx_u(tex_pos)].albedo_and_origin_dist.xyz, 1.);

    //return vec4<f32>(TINDEX_COLORS[geometry_buffer[tex_pos.x + tex_pos.y * uniforms.width].triangle_index], 1.);

    let geometry = collect_geo_u(tex_pos);
    let old_geometry = collect_geo_t2d(tex_pos, texture_geo_pack_0_old, texture_geo_pack_1_old);

    let t_sim = (clamp(geometry.similarity_score, -1., 3.) + 1.) / 4.;
    
    switch uniforms.visualisation_mode {
        case 0 : { return vec4<f32>(aces_film(textureLoad(texture_rt, tex_pos, 0).xyz), 1.); }        // rt
        //case 1 : { return vec4<f32>(vec3<f32>(geometry.variance / 1.), 1.); }        // variance
        case 2 : { return vec4<f32>(aces_film(textureLoad(texture_denoise_0, tex_pos, 0).xyz), 1.); } // denoise 0
        case 3 : { return vec4<f32>(aces_film(textureLoad(texture_denoise_1, tex_pos, 0).xyz), 1.); } // denoise 1
        case 4 : { return vec4<f32>(aces_film(textureLoad(texture_rt, tex_pos, 0).xyz * geometry.albedo), 1.); }        // rt * albedo
        case 5 : { return vec4<f32>(aces_film(textureLoad(texture_denoise_0, tex_pos, 0).xyz * geometry.albedo), 1.); } // denoise 0 * albedo
        case 6 : { return vec4<f32>(aces_film(textureLoad(texture_denoise_1, tex_pos, 0).xyz * geometry.albedo), 1.); } // denoise 1 * albedo
        case 7 : { return vec4<f32>(geometry.albedo, 1.); }                                // albedo
        case 8 : { return vec4<f32>(geometry.normal / 1.5, 1.); }                          // normal
        case 9 : { return vec4<f32>(abs(geometry.normal) / 1.5, 1.); }                     // abs normal
        case 10: { return vec4<f32>(vec3<f32>(geometry.depth), 1.); }                      // depth
        case 11: { return vec4<f32>(geometry.position / 2., 1.); }                        // scene location
        case 12: { return vec4<f32>(abs(geometry.position) / 2., 1.); }                   // abs scene location
        case 13: { return vec4<f32>(vec3<f32>(geometry.distance_from_origin / 50.), 1.); } // dist from origin
        case 14: { return vec4<f32>(get_tindex_color(geometry.object_index), 1.); }        // object index
        case 15: { return select(vec4<f32>(0., 1., 0., 1.), vec4<f32>(1., 0., 0., 1.), geometry.was_invalidated); }        // invalidations
        case 1: { return vec4<f32>(0., 1., 0., 1.) * t_sim + vec4<f32>(1., 0., 0., 1.) * (1. - t_sim); }        // invalidations
        default: { return vec4<f32>(0., 0., 0., 1.); }
    }
}
