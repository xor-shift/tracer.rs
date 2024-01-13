use super::*;

pub struct PerfectDielectric {
    pub index_of_refraction: Float,
}

impl Material for PerfectDielectric {
    fn interact<Gen: stuff::rng::UniformRandomBitGenerator>(&self, intersection: &Intersection, gen: &mut Gen) -> Interaction {
        let (transmittant_index, incident_index) = if intersection.going_in() { (self.index_of_refraction, 1.) } else { (1., self.index_of_refraction) };

        let (wi, weight) = if let Some((refraction, p_reflection)) = ray::refract(intersection.wo, intersection.oriented_normal(), incident_index, transmittant_index) {
            let generated = f64::generate_canonical(gen).get();

            // the brdf and the pdf are the same, they hence cancel eachother
            if generated < p_reflection {
                (ray::reflect(intersection.wo, intersection.oriented_normal()), 1.)
            } else {
                (refraction, 1.)
            }
        } else {
            (ray::reflect(intersection.wo, intersection.oriented_normal()), 1.)
        };

        Interaction {
            normal: intersection.normal,
            wi,
            wo: intersection.wo,

            weight,
            attenuation: color::LinearRGB(Vec3::new_explode(0.999)),
            emittance: color::LinearRGB(Vec3::new_explode(0.)),
        }
    }
}
