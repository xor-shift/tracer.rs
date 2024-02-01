struct Sphere {
    center: vec3<f32>,
    radius: f32,
}

// value written to out is indeterminate if the function returns false.
fn solve_sphere_quadratic(a: f32, b: f32, c: f32, out: ptr<function, f32>) -> bool {
    let delta = b * b - 4. * a * c;
    let sqrt_delta = sqrt(delta);

    let raw_sol_0 = (-b + sqrt_delta) / (2. * a);
    let raw_sol_1 = (-b - sqrt_delta) / (2. * a);

    let sol_0 = select(raw_sol_0, 1. / 0., raw_sol_0 >= 0.);
    let sol_1 = select(raw_sol_1, 1. / 0., raw_sol_1 >= 0.);

    let solution = select(sol_0, sol_1, sol_0 < sol_1);

    *out = solution;

    return delta >= 0.;
}

fn sphere_intersect(sphere: Sphere, ray: Ray, out: ptr<function, Intersection>) -> bool {
    let direction = ray.origin - sphere.center;

    let a = 1.;
    let b = 2. * dot(direction, ray.direction);
    let c = dot(direction, direction) - (sphere.radius * sphere.radius);

    var t = 0.;
    if !solve_sphere_quadratic(a, b, c, &t) {
        return false;
    }

    *out = Intersection(
        t,
        ray.origin + ray.direction * t,
        vec3<f32>(0.),
        -ray.direction,
    );

    return true;
}
