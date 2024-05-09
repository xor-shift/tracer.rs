struct Box {
    min: vec3<f32>,
    max: vec3<f32>,

    material: Material,
}

// https://tavianator.com/2011/ray_box.html
fn _box_intersect_impl_baseline(box: Box, ray: Ray, inout_distance: ptr<function, f32>) -> bool {
    var tmin = 0.;
    var tmax = *inout_distance;

    for (var i = 0; i < 3; i++) {
        let t1 = (box.min[i] - ray.origin[i]) * ray.direction_reciprocal[i];
        let t2 = (box.max[i] - ray.origin[i]) * ray.direction_reciprocal[i];

        tmin = max(tmin, min(t1, t2));
        tmax = min(tmax, max(t1, t2));
    }

    let intersected = tmin < tmax;

    if intersected {
        *inout_distance = tmin;
    }

    return intersected;
}

// https://tavianator.com/2015/ray_box_nan.html
fn _box_intersect_impl_exclusive(box: Box, ray: Ray, inout_distance: ptr<function, f32>) -> bool {
    var tmin = 0.;
    var tmax = *inout_distance;

    for (var i = 0; i < 3; i++) {
        let t1 = (box.min[i] - ray.origin[i]) * ray.direction_reciprocal[i];
        let t2 = (box.max[i] - ray.origin[i]) * ray.direction_reciprocal[i];

        tmin = max(tmin, min(min(t1, t2), tmax));
        tmax = min(tmax, max(max(t1, t2), tmin));
    }

    let intersected = tmin < tmax;

    if intersected {
        *inout_distance = tmin;
    }

    return intersected;
}

// https://tavianator.com/2022/ray_box_boundary.html#boundaries
fn _box_intersect_impl_inclusive(box: Box, ray: Ray, inout_distance: ptr<function, f32>) -> bool {
    var tmin = 0.;
    var tmax = *inout_distance;

    for (var i = 0; i < 3; i++) {
        let t1 = (box.min[i] - ray.origin[i]) * ray.direction_reciprocal[i];
        let t2 = (box.max[i] - ray.origin[i]) * ray.direction_reciprocal[i];

        tmin = min(max(t1, tmin), max(t2, tmin));
        tmax = max(min(t1, tmax), min(t2, tmax));
    }

    let intersected = tmin <= tmax;

    if intersected {
        *inout_distance = tmin;
    }

    return intersected;
}

// https://tavianator.com/2022/ray_box_boundary.html#boundaries
fn _box_intersect_impl_signs(box: Box, ray: Ray, inout_distance: ptr<function, f32>) -> bool {
    var tmin = 0.;
    var tmax = *inout_distance;

    for (var i = 0; i < 3; i++) {
        let sign_bit = select(0, 1, ray.direction_reciprocal[i] < 0);

        let bmin = select(box.min[i], box.max[i], sign_bit == 1);
        let bmax = select(box.min[i], box.max[i], sign_bit == 0);

        let dmin = (bmin - ray.origin[i]) * ray.direction_reciprocal[i];
        let dmax = (bmax - ray.origin[i]) * ray.direction_reciprocal[i];

        tmin = max(dmin, tmin);
        tmax = min(dmax, tmax);
    }

    let intersected = tmin <= tmax;

    if intersected {
        *inout_distance = tmin;
    }

    return intersected;
}

fn _box_intersect_impl_another(box: Box, ray: Ray, inout_distance: ptr<function, f32>) -> bool {
    let t0 = (box.min - ray.origin) * ray.direction_reciprocal;
    let t1 = (box.max - ray.origin) * ray.direction_reciprocal;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);

    let max_of_min = max(max(tmin.x, tmin.y), tmin.z);
    let min_of_max = min(min(min(tmax.x, tmax.y), tmax.z), *inout_distance);

    let ret = max_of_min <= min_of_max;

    if ret {
        *inout_distance = max_of_min;
    }

    return ret;
}

fn _box_intersect(box: Box, ray: Ray, inout_distance: ptr<function, f32>) -> bool {
    return _box_intersect_impl_inclusive(box, ray, inout_distance);
}

fn _box_normal_naive(box: Box, global_pt: vec3<f32>) -> vec3<f32> {
    /*

    +---+ (4, 3)
    |   x (4, 2)
    y   | (0, 1)
    +---+

    global -> local     -> per-axis norm -> fudge       -> trunc
    (4, 2) -> (2, .5)   -> (1, .3)       -> (1.1, .4)   -> (1, 0)
    (0, 1) -> (-2, -.5) -> (-1, -.3)     -> (-1.1, -.4) -> (-1, 0)

    */

    let center = (box.min + box.max) * 0.5;
    let local_pt = global_pt - center;

    let half_side_lengths = (box.max - box.min) * .5;
    let norm_local_pt = local_pt / half_side_lengths;
    let fudged = norm_local_pt * 1.0001;

    return normalize(trunc(fudged));
}

fn _box_normal(box: Box, global_pt: vec3<f32>) -> vec3<f32> {
    return _box_normal_naive(box, global_pt);
}

fn box_intersect_pt0(box: Box, ray: Ray, inout_intersection: ptr<function, Intersection>) -> bool {
    var t = (*inout_intersection).t;
    if !_box_intersect(box, ray, &t) {
        return false;
    }

    (*inout_intersection).t = t;

    return true;
}

fn box_intersect_pt1(box: Box, ray: Ray, inout_intersection: ptr<function, Intersection>) {
    var t = (*inout_intersection).t;
    let global_pt = ray.origin + t * ray.direction;
    let normal = _box_normal(box, global_pt);

    *inout_intersection = Intersection (
        /* material  */ box.material,

        /* wo        */ -ray.direction,
        /* t         */ t,

        /* global_pt */ global_pt,
        /* normal    */ normal,
        /* uv        */ vec2<f32>(0.),
    );
}

fn box_intersect(box: Box, ray: Ray, inout_intersection: ptr<function, Intersection>) -> bool {
    var t = (*inout_intersection).t;
    if !_box_intersect(box, ray, &t) {
        return false;
    }

    let global_pt = ray.origin + t * ray.direction;
    let normal = _box_normal(box, global_pt);

    *inout_intersection = Intersection (
        /* material  */ box.material,

        /* wo        */ -ray.direction,
        /* t         */ t,

        /* global_pt */ global_pt,
        /* normal    */ normal,
        /* uv        */ vec2<f32>(0.),
    );

    return true;
}
