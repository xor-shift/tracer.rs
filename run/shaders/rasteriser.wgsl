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
