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
    //@location(3) old_scene_position: vec3<f32>,
    @location(3) triangle_index: u32,
}

@group(0) @binding(0) var<uniform> uniforms: State;
@group(0) @binding(1) var<uniform> uniforms_old: State; // retarded
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
        vertex_index / 3u + 1u,
    );
}

struct FragmentOutput {
    @location(0) pack_0: vec4<u32>,
    @location(1) pack_1: vec4<u32>,
}

@fragment fn fs_main(
    in: VertexOutput,
) -> FragmentOutput {
    let test = in.position.xyz / in.position.w;
    let pixel = vec2<u32>(trunc(in.position.xy));

    let prev = textureLoad(previous_frame_pt, pixel, 0);
    let integrated = textureLoad(integrated_frame_pt, pixel, 0);

    let geo = GeometryElement(
        /* albedo   */ in.color,
        /* variance */ 0.,
        /* normal   */ in.normal,
        /* depth    */ in.position.z,
        /* position */ in.scene_position,
        /* distance */ length(in.scene_position),
        /* index    */ in.triangle_index,
        /* inval'd  */ false, // to be filled in by the path tracer
        0.,
    );

    let packed_geo = pack_geo(geo);

    return FragmentOutput(
        packed_geo.pack_0,
        packed_geo.pack_1,
    );
}
