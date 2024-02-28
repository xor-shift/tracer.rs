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
