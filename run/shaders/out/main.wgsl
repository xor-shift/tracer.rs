struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

struct MainUniform {
    // at frame no 0, texture 1 should be used and texture 0 should be drawn on
    frame_no: u32,
    current_instant: f32,
    seed_0: u32,
    seed_1: u32,
    seed_2: u32,
    seed_3: u32,
    visualisation_mode: i32,
}

struct GeometryElement {
    normal_and_depth: vec4<f32>,
    albedo: vec4<f32>,
    position: vec4<f32>,
}

fn gb_idx_i(coords: vec2<i32>) -> i32 {
    let cols = textureDimensions(texture_rt).x;
    return coords.x + coords.y * i32(cols);
}

fn gb_idx_u(coords: vec2<u32>) -> u32 {
    let cols = textureDimensions(texture_rt).x;
    return coords.x + coords.y * cols;
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
@vertex fn vs_main(vert: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = vert.tex_coords;
    out.position = vec4<f32>(vert.position, 1.0);
    return out;
}

@group(0) @binding(0) var<uniform> uniforms: MainUniform;

@group(1) @binding(0) var texture_rt: texture_2d<f32>;
@group(1) @binding(1) var<storage, read> geometry_buffer: array<GeometryElement>;
@group(1) @binding(2) var texture_denoise_0: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(3) var texture_denoise_1: texture_storage_2d<rgba8unorm, read_write>;

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

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = textureDimensions(texture_rt);
    let tex_pos = vec2<u32>(vec2<f32>(tex_size) * in.tex_coords);

    switch uniforms.visualisation_mode {
        // indirect light
        case 0 : { return textureLoad(texture_rt, tex_pos, 0); }
        // indirect light composited with albedo
        case 1 : { return vec4<f32>(textureLoad(texture_rt, tex_pos, 0).xyz * geometry_buffer[gb_idx_u(tex_pos)].albedo.xyz, 1.); }

        // filtered indirect light composited with albedo
        case 2 : { return vec4<f32>(textureLoad(texture_denoise_0, tex_pos).xyz * geometry_buffer[gb_idx_u(tex_pos)].albedo.xyz, 1.); }
        case 3 : { return vec4<f32>(textureLoad(texture_denoise_1, tex_pos).xyz * geometry_buffer[gb_idx_u(tex_pos)].albedo.xyz, 1.); }

        // albedo
        case 4 : { return vec4<f32>(geometry_buffer[gb_idx_u(tex_pos)].albedo.xyz, 1.); }

        // normals, absolute normals
        case 5 : { return vec4<f32>(geometry_buffer[gb_idx_u(tex_pos)].normal_and_depth.xyz, 1.); }
        case 6 : { return vec4<f32>(abs(geometry_buffer[gb_idx_u(tex_pos)].normal_and_depth.xyz), 1.); }

        //position
        case 7 : { return vec4<f32>(geometry_buffer[gb_idx_u(tex_pos)].position.xyz / 10., 1.); }

        //depth
        case 8 : { return vec4<f32>(vec3<f32>((geometry_buffer[gb_idx_u(tex_pos)].normal_and_depth.w - 10.) / 10.), 1.); }

        default: { return vec4<f32>(0., 0., 0., 1.); }
    }
}
