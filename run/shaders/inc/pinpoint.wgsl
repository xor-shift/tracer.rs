struct PinpointCamera {
    fov: f32,
}

fn pinpoint_generate_ray(
    camera: PinpointCamera,
    screen_coords: vec2<u32>,
    screen_dims: vec2<u32>,
    pos: vec3<f32>,
) -> Ray {
    let half_theta = camera.fov / 2.;
    let d = (1. / (2. * sin(half_theta))) * sqrt(abs((f32(screen_dims.x) * (2. - f32(screen_dims.x)))));

    //let offset = vec2<f32>(rand() * 2. - 1., rand() * 2. - 1.);

    let direction = normalize(vec3<f32>(
        f32(screen_coords.x) - (f32(screen_dims.x) / 2.),
        f32(screen_dims.y - screen_coords.y - 1u) - f32(screen_dims.y) / 2.,
        d,
    ));

    return Ray(pos, direction);
}
