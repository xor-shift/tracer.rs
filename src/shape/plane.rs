use crate::*;
use types::*;

pub struct Plane {
    pub center: Vec3,
    pub normal: REVec3,
    pub uv_scale: Float,
}

impl Plane {
    fn isect_step_0(&self, ray: &crate::Ray) -> Option<(Float, Float)> {
        let divisor = self.normal.0.dot(ray.direction.0);
        let divident = (self.center - ray.origin).dot(self.normal.0);

        if divisor.is_sign_negative() != divident.is_sign_negative() {
            None
        } else {
            Some((divident, divisor))
        }
    }
}

impl Intersectable for Plane {
    fn hit_check(&self, ray: &crate::ray::Ray) -> bool {
        let divisor = self.normal.0.dot(ray.direction.0);
        let divident = (self.center - ray.origin).dot(self.normal.0);

        divisor.is_sign_negative() == divident.is_sign_negative()
    }

    fn intersect(&self, ray: &crate::ray::Ray, _reqs: IntersectionRequirements, previous: &Option<Intersection>) -> Option<Intersection> {
        let denom = self.normal.0.dot(ray.direction.0);
        let t = (self.center - ray.origin).dot(self.normal.0) / denom;

        if t < 0.0001 || t > previous.as_ref().map(|v| v.distance).unwrap_or(Float::INFINITY) {
            return None;
        }

        let (dp_du, dp_dv) = super::dummy_dpduv(self.normal);

        if cfg!(debug_assertions) {
            let got_norm = dp_du.0.cross(dp_dv.0);
            assert_eq!(got_norm, self.normal.0);
        }

        Some(Intersection {
            position: ray.origin + ray.direction.0 * t,
            distance: t,
            wo: REVec3(-ray.direction.0),

            normal: self.normal,
            texture_coords: VecUV(Vec2::new_explode(0.)),
            dp_du: REVec3(dp_du.0 * self.uv_scale),
            dp_dv: REVec3(dp_dv.0 * self.uv_scale),
            reflection_to_surface: orthonormal_from_xz(dp_du.0, self.normal.0),

            material_id: 0,
        })
    }
}

impl Shape for Plane {
    fn sample_surface<Gen: stuff::rng::UniformRandomBitGenerator>(&self, gen: &mut Gen) -> SurfaceSample { todo!() }

    fn surface_area(&self) -> Float { Float::INFINITY }
    fn volume(&self) -> Float { 0. }

    fn center(&self) -> Vec3 { self.center }
    fn global_bounds(&self) -> Extent3D { Extent3D::new_infinite() }
}
