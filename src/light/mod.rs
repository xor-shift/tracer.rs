use crate::*;
use types::*;

pub struct Skybox {}

impl Intersectable for Skybox {
    fn hit_check(&self, ray: &crate::ray::Ray) -> bool { true }

    fn intersect(&self, ray: &crate::ray::Ray, _reqs: IntersectionRequirements, previous: &Option<Intersection>) -> Option<Intersection> {
        let faux_pos = SHVec3(-ray.direction.0);
        let uv = shape::sphere_uv(faux_pos, 1.);
        let surface_params = sphere_surface_params(faux_pos, 1., uv);

        Some(Intersection {
            position: Vec3::new_explode(Float::INFINITY),
            distance: Float::INFINITY,
            wo: REVec3(-ray.direction.0),

            normal: ray.direction,

            texture_coords: uv,

            dp_du: surface_params[0],
            dp_dv: surface_params[1],
            reflection_to_surface: orthonormal_from_xz(surface_params[0].0, ray.direction.0),

            material_id: 0,
        })
    }
}

impl Shape for Skybox {
    fn sample_surface<Gen: stuff::rng::UniformRandomBitGenerator>(&self, gen: &mut Gen) -> SurfaceSample { todo!() }

    fn surface_area(&self) -> Float { Float::INFINITY }
    fn volume(&self) -> Float { Float::INFINITY }

    fn center(&self) -> Vec3 { Vec3::new_explode(0.) }
    fn global_bounds(&self) -> Extent3D { Extent3D::new_infinite() }
}
