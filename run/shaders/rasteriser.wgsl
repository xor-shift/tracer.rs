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

@group(0) @binding(0) var<uniform> uniforms: RasteriserUniform;
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
        uniforms.camera * vec4<f32>(vert.position, 1.),
        MATERIAL_COLORS[vert.material],
        vert.normal,
        vert.position,
        vertex_index / 3u,
    );
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

struct FragmentOutput {
    @location(0) albdeo: vec4<f32>,
    @location(1) pack_normal_depth: vec4<f32>,
    @location(2) pack_positon_distance: vec4<f32>,
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
    );

    /*var out: FragmentOutput;
    out.albedo = vec4<f32>(in.color, 1.);
    out.pack_positon_distance = vec4<f32>(in.scene_position, length(in.scene_position));

    //if geometry_buffer[pixel.x + pixel.y * uniforms.width].normal_and_depth.w <= in.position.w {
        /*geometry_buffer[pixel.x + pixel.y * uniforms.width].normal_and_depth = vec4<f32>(in.normal, test.z);
        geometry_buffer[pixel.x + pixel.y * uniforms.width].albedo_and_origin_dist = vec4<f32>(in.color, length(in.scene_position));
        geometry_buffer[pixel.x + pixel.y * uniforms.width].direct_illum = vec3<f32>(0.);
        geometry_buffer[pixel.x + pixel.y * uniforms.width].scene_position = in.scene_position;
        geometry_buffer[pixel.x + pixel.y * uniforms.width].triangle_index = in.triangle_index;*/
    //}

    //return vec4<f32>(1., 0., 0., 1.);
    //return vec4<f32>(geometry_buffer[pixel.x - 1 + pixel.y * uniforms.width].albedo_and_origin_dist.xyz, 1.);
    //return vec4<f32>(vec2<f32>(pixel.xy) / vec2<f32>(f32(uniforms.width), f32(uniforms.height)), 0., 1.);
    //return vec4<f32>(vec3<f32>(in.position.z), 1.);
    //return vec4<f32>(TINDEX_COLORS[in.triangle_index % 7u], 1.);

    return out;*/
}
