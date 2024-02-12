use super::*;

pub struct PerfectMirror {}

impl Material for PerfectMirror {
    fn interact<Gen: stuff::rng::UniformRandomBitGenerator>(&self, intersection: &Intersection, gen: &mut Gen) -> Interaction {
        let wi = ray::reflect(intersection.wo, intersection.normal);

        Interaction {
            normal: intersection.normal,
            wi,
            wo: intersection.wo,

            weight: 1.,
            attenuation: color::LinearRGB(Vec3::new_explode(0.999)),
            emittance: color::LinearRGB(Vec3::new_explode(0.)),
        }
    }
}
