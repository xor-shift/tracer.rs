struct Sphere {
    center: vec3<f32>,
    radius: f32,
    material: u32,
}

// value written to out is indeterminate if the function returns false.
fn solve_sphere_quadratic(a: f32, b: f32, c: f32, out: ptr<function, f32>) -> bool {
    let delta = b * b - 4. * a * c;
    let sqrt_delta = sqrt(delta);

    let raw_sol_0 = (-b + sqrt_delta) / (2. * a);
    let raw_sol_1 = (-b - sqrt_delta) / (2. * a);

    // nans should become infs here
    // i don't know actually, is nan ordered in wgsl? does wgsl even have a nan?
    let sol_0 = select(INF, raw_sol_0, raw_sol_0 >= 0.);
    let sol_1 = select(INF, raw_sol_1, raw_sol_1 >= 0.);

    let solution = select(sol_1, sol_0, sol_0 < sol_1);

    *out = solution;

    return delta >= 0.;
}

fn sphere_uv(local_point: vec3<f32>, radius: f32) -> vec2<f32> {
    let θ_uncorrected = atan2(local_point[1], local_point[0]);
    let θ = select(θ_uncorrected, θ_uncorrected + 2. * PI, θ_uncorrected < 0.);

    let φ = PI - acos(local_point[2] / radius);

    let u = θ * 0.5 * FRAC_1_PI;
    let v = φ * FRAC_1_PI;

    return vec2<f32>(u, v);
}

fn sphere_surface_params(local_point: vec3<f32>, radius: f32, uv: vec2<f32>) -> array<vec3<f32>, 2> {
    let π = PI;
    let x = local_point.x;
    let y = local_point.y;

    let θ = uv[0] * 2. * π;
    let φ = uv[1] * π;

    let sinθ = sin(θ);
    let cosθ = cos(θ);

    let δxδu = -2. * π * y;
    let δyδu = 2. * π * x;
    let δzδu = 0.;

    let δxδv = 2. * π * cosθ;
    let δyδv = 2. * π * sinθ;
    let δzδv = -radius * π * sin(φ);

    return array<vec3<f32>, 2>(
        vec3<f32>(δxδu, δyδu, δzδu),
        vec3<f32>(δxδv, δyδv, δzδv)
    );
}

fn sphere_intersect(sphere: Sphere, ray: Ray, best: f32, out: ptr<function, Intersection>) -> bool {
    let direction = ray.origin - sphere.center;

    let a = 1.;
    let b = 2. * dot(direction, ray.direction);
    let c = dot(direction, direction) - (sphere.radius * sphere.radius);

    var t = 0.;
    if !solve_sphere_quadratic(a, b, c, &t) {
        return false;
    }

    let global_position = ray.origin + ray.direction * t;
    let local_position = global_position - sphere.center;
    let normal = normalize(local_position);
    let oriented_normal = select(-normal, normal, dot(ray.direction, normal) < 0.);
    let uv = sphere_uv(local_position, sphere.radius);

    //let surface_params = sphere_surface_params(local_position, sphere.radius, uv);
    //let refl_to_surface = orthonormal_from_xz(normalize(surface_params[0]), oriented_normal);

    *out = Intersection(
        t,
        global_position,
        normal,
        -ray.direction,
        sphere.material,
        //refl_to_surface,
    );

    return true;
}

fn sphere_sample(sphere: Sphere) -> SurfaceSample {
    var pdf: f32;
    let normal = sample_sphere_3d(&pdf);
    pdf /= sphere.radius * sphere.radius;

    let local_position = normal * sphere.radius;
    let global_position = local_position + sphere.center;

    return SurfaceSample(
        global_position,
        sphere_uv(local_position, sphere.radius),
        normal,
        pdf,
    );
}
