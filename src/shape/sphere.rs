use crate::*;
use stuff::{FloatConstants, FloatingPoint};

enum QuadraticSolution<T> {
    None,
    SingleRoot(T),
    TwoRoots(T, T),
}

#[allow(non_snake_case)]
fn solve_quadratic<T: FloatingPoint>(a: T, b: T, c: T) -> QuadraticSolution<T> {
    let two = T::as_from(2);

    let Δ = b * b - T::as_from(4) * a * c;

    if Δ < T::zero() {
        QuadraticSolution::None
    } else if Δ == T::zero() {
        QuadraticSolution::SingleRoot(-b / (two * a))
    } else {
        let sqrt_Δ = Δ.sqrt();
        QuadraticSolution::TwoRoots((-b + sqrt_Δ) / (two * a), (-b - sqrt_Δ) / (two * a))
    }
}

#[debug_requires(is_normalised(point.0 / radius))]
#[debug_ensures(ret.is_ok())]
fn sphere_uv(point: SHVec3, radius: Float) -> VecUV {
    let π = <Float as FloatConstants>::PI;
    let invπ = <Float as FloatConstants>::FRAC_1_PI;

    let θ = Float::atan2(point.0[1], point.0[0]);
    let θ = if θ < 0. { θ + 2. * π } else { θ };

    let φ = π - (point.0[2] / radius).acos();

    let u = θ * 0.5 * invπ;
    let v = φ * invπ;

    VecUV(Vec2::new([u, v]))
}

#[debug_requires(is_normalised(point.0 / radius))]
#[debug_requires(uv.is_ok())]
fn sphere_surface_params(point: SHVec3, radius: Float, uv: VecUV) -> [REVec3; 2] {
    let π = <Float as FloatConstants>::PI;
    let [x, y, _z] = point.0 .0;

    let θ = uv.0[0] * 2. * π;
    let φ = uv.0[1] * π;

    debug_assert!(-0.001 <= θ && θ <= 2.001 * π);
    debug_assert!(-0.001 <= φ && φ <= 1.001 * π);

    let (sinθ, cosθ) = θ.sin_cos();

    let δxδu = -2. * π * y;
    let δyδu = 2. * π * x;
    let δzδu = 0.;

    let δxδv = 2. * π * cosθ;
    let δyδv = 2. * π * sinθ;
    let δzδv = -radius * π * φ.sin();

    [REVec3(Vec3::new([δxδu, δyδu, δzδu])), REVec3(Vec3::new([δxδv, δyδv, δzδv]))]
}

pub struct Sphere {
    pub center: Vec3,
    pub radius: Float,
}

impl Sphere {
    fn get_t(&self, ray: &Ray) -> Option<Float> {
        let direction = ray.origin - self.center;

        let a = 1.;
        let b = 2. * direction.dot(ray.direction.0);
        let c = direction.dot(direction) - (self.radius * self.radius);

        let threshold = 0.000000001;

        let t = match solve_quadratic(a, b, c) {
            QuadraticSolution::None => None,
            QuadraticSolution::SingleRoot(t) => {
                if t > threshold {
                    Some(t)
                } else {
                    None
                }
            }
            QuadraticSolution::TwoRoots(t_0, t_1) => match (t_0 > threshold, t_1 > threshold) {
                (false, false) => None,
                (true, false) => Some(t_0),
                (false, true) => Some(t_1),
                (true, true) => Some(t_0.min(t_1)),
            },
        };

        t
    }
}

impl Intersectable for Sphere {
    fn hit_check(&self, ray: &crate::ray::Ray) -> bool { self.get_t(ray).is_some() }

    #[debug_requires(is_normalised(ray.direction.0))]
    #[debug_ensures(if let Some(v) = &ret { v.distance >= 0. } else { true })]
    fn intersect(&self, ray: &Ray, _reqs: IntersectionRequirements, previous: &Option<Intersection>) -> Option<Intersection> {
        self.get_t(ray)
            .and_then(|t| {
                if let Some(best) = previous
                    && best.distance < t
                {
                    None
                } else {
                    Some(t)
                }
            })
            .map(|t| {
                let pos = ray.origin + ray.direction.0 * t;
                let local_pos = SHVec3(pos - self.center);

                let normal = REVec3(local_pos.0 / self.radius);
                let uv = sphere_uv(local_pos, self.radius);
                let surface_params = sphere_surface_params(local_pos, self.radius, uv);

                if cfg!(debug_assertions) {
                    let norm_sp = [
                        surface_params[0].0.normalized(), //
                        surface_params[1].0.normalized(),
                    ];

                    assert!((norm_sp[0].dot(norm_sp[1])).abs() <= 0.001);

                    let got_norm = norm_sp[0].cross(norm_sp[1]);

                    assert!(vectors_are_same(got_norm, normal.0));
                }

                Intersection {
                    distance: t,
                    position: pos,
                    wo: REVec3(-ray.direction.0),

                    normal,
                    dp_du: surface_params[0],
                    dp_dv: surface_params[1],
                    reflection_to_surface: orthonormal_from_xz(surface_params[0].0.normalized(), normal.0),

                    material_id: 0,
                    texture_coords: uv,
                }
            })
    }
}

impl Shape for Sphere {
    fn sample_surface<Gen: stuff::rng::UniformRandomBitGenerator>(&self, gen: &mut Gen) -> SurfaceSample {
        let (sample, pdf) = stuff::rng::distributions::sphere::UniformSphereSampler::new().sample(gen);
        let sample = Vec3::new(sample);

        todo!(); //
    }

    fn surface_area(&self) -> Float { <Float as FloatConstants>::PI * 2. * self.radius }
    fn volume(&self) -> Float { <Float as FloatConstants>::PI * self.radius * self.radius }

    fn center(&self) -> Vec3 { self.center }
    fn global_bounds(&self) -> Extent3D { Extent3D(self.center - Vec3::new_explode(self.radius), self.center + Vec3::new_explode(self.radius)) }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sphere_intersection() {
        let sphere = Sphere {
            center: Vec3::new([1., 1., 5.]),
            radius: 1.5,
        };

        assert!(sphere
            .intersect(&Ray::new(Vec3::new([-1., -1., 0.]), REVec3(Vec3::new([0., 0., 1.]).normalized())), IntersectionRequirements::empty(), &None)
            .is_none());

        assert!(sphere
            .intersect(&Ray::new(Vec3::new([1., 1., 0.]), REVec3(Vec3::new([0., 0., 1.]).normalized())), IntersectionRequirements::empty(), &None)
            .is_some());

        assert!(sphere
            .intersect(&Ray::new(Vec3::new([3., 3., 0.]), REVec3(Vec3::new([0., 0., 1.]).normalized())), IntersectionRequirements::empty(), &None)
            .is_none());
    }
}
