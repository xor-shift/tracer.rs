use super::*;

pub struct Lambertian {
    pub albedo: color::LinearRGB,
    pub emittance: color::LinearRGB,
}

impl Material for Lambertian {
    fn interact<Gen: stuff::rng::UniformRandomBitGenerator>(&self, intersection: &Intersection, gen: &mut Gen) -> Interaction {
        let (wi, wi_pdf) = stuff::rng::distributions::sphere::UniformSphereSampler::new().sample(gen);
        let wi = Vec3::new(wi);
        let wi = REVec3(if wi.dot(intersection.normal.0) >= 0. { wi } else { -wi });
        let wi_pdf = wi_pdf * 2.;
        let brdf = 0.5 / std::f64::consts::PI;
        let weight = wi.0.dot(intersection.normal.0).abs() * brdf / wi_pdf;

        Interaction {
            normal: intersection.normal,
            wi,
            wo: intersection.wo,

            weight,
            attenuation: self.albedo,
            emittance: self.emittance,
        }
    }
}

pub struct LambertianIS {
    pub albedo: color::LinearRGB,
    pub emittance: color::LinearRGB,
}

impl Material for LambertianIS {
    fn interact<Gen: stuff::rng::UniformRandomBitGenerator>(&self, intersection: &Intersection, gen: &mut Gen) -> Interaction {
        let (wi_data, wi_pdf) = stuff::rng::distributions::sphere::CosineWeightedHemisphereSampler::new().sample(gen);
        let wi_original = SVec3(Vec3::new(wi_data));

        debug_assert!(is_normalised(wi_original.0));

        let wi = intersection.convert_to_reflection_space(wi_original);
        let wi = REVec3(if intersection.going_in() { wi.0 } else { -wi.0 });

        debug_assert!(is_normalised(wi.0));

        let brdf = 0.5 / std::f64::consts::PI;
        let weight = wi.0.dot(intersection.normal.0).abs() * brdf / wi_pdf;

        Interaction {
            normal: intersection.normal,
            wi,
            wo: intersection.wo,

            weight,
            attenuation: self.albedo,
            emittance: self.emittance,
        }
    }
}
