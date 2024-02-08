struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

fn orthonormal_from_xz(x: vec3<f32>, z: vec3<f32>) -> mat3x3<f32> {
    let y = cross(z, x);

    return mat3x3<f32>(
        x[0], x[1], x[2],
        y[0], y[1], y[2],
        z[0], z[1], z[2],
    );
}

// https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
fn rodrigues(x: vec3<f32>, norm_from: vec3<f32>, norm_to: vec3<f32>) -> vec3<f32> {
    let along = normalize(cross(norm_from, norm_to));
    let cosφ = dot(norm_from, norm_to);
    let sinφ = length(cross(norm_from, norm_to));
    if abs(cosφ) > 0.999 {
        return select(x, -x, cosφ < 0.);
    } else {
        return x * cosφ + cross(along, x) * sinφ + along * dot(along, x) * (1. - cosφ);
    }
}

// only for reflection space to surface space conversions
fn rodrigues_fast(x: vec3<f32>, norm_to: vec3<f32>) -> vec3<f32> {
    let cosφ = norm_to.z;
    let sinφ = sqrt(norm_to.y * norm_to.y + norm_to.x * norm_to.x);

    let along = vec3<f32>(-norm_to.y, norm_to.x, 0.) / sinφ;

    if abs(cosφ) > 0.999 {
        return select(x, -x, cosφ < 0.);
    } else {
        return x * cosφ + cross(along, x) * sinφ + along * dot(along, x) * (1. - cosφ);
    }
}

struct Intersection {
    distance: f32,
    position: vec3<f32>,
    normal: vec3<f32>,
    wo: vec3<f32>,
    material_idx: u32,

    //refl_to_surface: mat3x3<f32>,
}

fn dummy_intersection(ray: Ray) -> Intersection {
    let inf = 1. / 0.;
    return Intersection(inf, vec3<f32>(inf), -ray.direction, -ray.direction, 0u);
}

fn intersection_going_in(intersection: Intersection) -> bool { return 0. < dot(intersection.wo, intersection.normal); }
fn intersection_oriented_normal(intersection: Intersection) -> vec3<f32> { return select(-intersection.normal, intersection.normal, intersection_going_in(intersection)); }
fn intersection_cos_theta_o(intersection: Intersection) -> f32 { return abs(dot(intersection.wo, intersection.normal)); }

fn pick_intersection(lhs: Intersection, rhs: Intersection) -> Intersection {
    if lhs.distance > rhs.distance {
        return rhs;
    } else {
        return lhs;
    }
}

fn reflect(wo: vec3<f32>, oriented_normal: vec3<f32>) -> vec3<f32> {
    return -wo + oriented_normal * 2. * dot(oriented_normal, wo);
}

fn schlick(cosθ: f32, η1: f32, η2: f32) -> f32 {
    let sqrt_r0 = ((η1 - η2) / (η1 + η2));
    let r0 = sqrt_r0 * sqrt_r0;
    let r = r0 + (1. - r0) * pow(1. - cosθ, 5.);

    return r;
}

fn refract(
    wo: vec3<f32>,
    normal: vec3<f32>,
    incident_index: f32,
    transmittant_index: f32,
    out_probability_reflection: ptr<function, f32>,
    out_refraction: ptr<function, vec3<f32>>,
) -> bool {
    let l = -wo;

    let index_ratio = incident_index / transmittant_index;

    let cosθ_i = dot(-l, normal);
    let sin2θ_i = 1. - cosθ_i * cosθ_i;
    let sin2θ_t = index_ratio * index_ratio * sin2θ_i;

    if sin2θ_t >= 1. {
        return false;
    }

    let cosθ_t = sqrt(1. - sin2θ_t);

    *out_probability_reflection = schlick(cosθ_i, incident_index, transmittant_index);
    *out_refraction = l * index_ratio + normal * (index_ratio * cosθ_i - cosθ_t);

    return true;
}

struct SurfaceSample {
    position: vec3<f32>,
    uv: vec2<f32>,
    normal: vec3<f32>,
    pdf: f32,
}
