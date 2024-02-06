struct Triangle {
    vertices: array<vec3<f32>, 3>,
    uv_offset: vec2<f32>,
    uv_scale: vec2<f32>,
    material: u32,
}

// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
fn triangle_intersect(triangle: Triangle, ray: Ray, best: f32, out: ptr<function, Intersection>) -> bool {
    let eps = 0.0001;

    let edge1 = triangle.vertices[1] - triangle.vertices[0];
    let edge2 = triangle.vertices[2] - triangle.vertices[0];
    let ray_cross_e2 = cross(ray.direction, edge2);
    let det = dot(edge1, ray_cross_e2);

    if det > -eps && det < eps {
        return false;
    }

    let inv_det = 1.0 / det;
    let s = ray.origin - triangle.vertices[0];
    let u = inv_det * dot(s, ray_cross_e2);

    if u < 0. || u > 1. {
        return false;
    }

    let s_cross_e1 = cross(s, edge1);
    let v = inv_det * dot(ray.direction, s_cross_e1);

    if v < 0. || u + v > 1. {
        return false;
    }

    let t = inv_det * dot(edge2, s_cross_e1);

    if t < eps || best < t {
        return false;
    }

    *out = Intersection(
        t,
        ray.origin + ray.direction * t,
        normalize(cross(edge1, edge2)),
        -ray.direction,
        triangle.material,
    );

    return true;
}
