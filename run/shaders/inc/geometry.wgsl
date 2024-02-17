/*struct GeometryElement {
    normal_and_depth: vec4<f32>,
    albedo_and_origin_dist: vec4<f32>,
    scene_position: vec3<f32>,
    triangle_index: u32,
}

fn ge_normal(ge: GeometryElement) -> vec3<f32> { return ge.normal_and_depth.xyz; }
fn ge_depth(ge: GeometryElement) -> f32 { return ge.normal_and_depth.w; }
fn ge_albedo(ge: GeometryElement) -> vec3<f32> { return ge.albedo_and_origin_dist.xyz; }
fn ge_origin_distance(ge: GeometryElement) -> f32 { return ge.albedo_and_origin_dist.w; }
//fn ge_position(ge: GeometryElement) -> vec3<f32> { return ge.position.xyz; }

fn gb_idx_i(coords: vec2<i32>) -> i32 {
    // let cols = textureDimensions(texture_rt).x;
    return coords.x + coords.y * i32(uniforms.width);
}

fn gb_idx_u(coords: vec2<u32>) -> u32 {
    // let cols = textureDimensions(texture_rt).x;
    return coords.x + coords.y * uniforms.width;
}*/

struct GeometryElement {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    depth: f32,
    position: vec3<f32>,
    distance_from_origin: f32,
    object_index: u32,
}

fn collect_geo_i(coords: vec2<i32>) -> GeometryElement {
    return collect_geo_u(vec2<u32>(max(coords, vec2<i32>(0))));
}
