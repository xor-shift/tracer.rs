use crate::*;
use stuff::qoi::Color;
use types::*;

mod dielectric;
mod lambertian;
mod mirror;

pub use dielectric::*;
pub use lambertian::*;
pub use mirror::*;

pub enum ColorSource {
    Normal,
    TextureCoords,
    Flat(LinearRGB),
}

impl ColorSource {
    pub fn get_color(&self, intersection: &Intersection) -> LinearRGB {
        match self {
            Self::Normal => LinearRGB(intersection.normal.0.abs()),
            Self::TextureCoords => LinearRGB(Vec3::new([
                intersection.texture_coords.0[0],
                intersection.texture_coords.0[1],
                0.,
            ])),
            Self::Flat(c) => *c,
        }
    }
}

pub trait Material {
    fn interact<Gen: stuff::rng::UniformRandomBitGenerator>(
        &self,
        intersection: &Intersection,
        gen: &mut Gen,
    ) -> Interaction;
}

pub enum EMaterial {
    LambertianDiffuse(lambertian::Lambertian),
    LambertianDiffuseIS(lambertian::LambertianIS),
    PerfectDielectric(dielectric::PerfectDielectric),
    PerfectMirror(mirror::PerfectMirror),
}

impl From<Lambertian> for EMaterial {
    fn from(value: Lambertian) -> Self {
        Self::LambertianDiffuse(value)
    }
}

impl From<LambertianIS> for EMaterial {
    fn from(value: LambertianIS) -> Self {
        Self::LambertianDiffuseIS(value)
    }
}

impl From<PerfectDielectric> for EMaterial {
    fn from(value: PerfectDielectric) -> Self {
        Self::PerfectDielectric(value)
    }
}

impl From<PerfectMirror> for EMaterial {
    fn from(value: PerfectMirror) -> Self {
        Self::PerfectMirror(value)
    }
}

impl Material for EMaterial {
    #[debug_requires(is_normalised(intersection.wo.0))]
    #[debug_requires(is_normalised(intersection.normal.0))]
    #[debug_ensures(is_normalised(ret.normal.0))]
    #[debug_ensures(is_normalised(ret.wi.0))]
    #[debug_ensures(is_normalised(ret.wo.0))]
    #[debug_ensures(ret.weight.is_finite())]
    fn interact<Gen: stuff::rng::UniformRandomBitGenerator>(
        &self,
        intersection: &Intersection,
        gen: &mut Gen,
    ) -> Interaction {
        match self {
            EMaterial::LambertianDiffuse(v) => v.interact(intersection, gen),
            EMaterial::LambertianDiffuseIS(v) => v.interact(intersection, gen),
            EMaterial::PerfectDielectric(v) => v.interact(intersection, gen),
            EMaterial::PerfectMirror(v) => v.interact(intersection, gen),
        }
    }
}
