use crate::*;
use types::*;

pub struct Ray {
    pub origin: Vec3,
    pub direction: REVec3,
    pub direction_reciprocals: Vec3,
}

impl Ray {
    #[debug_requires(is_normalised(direction.0))]
    pub fn new(origin: Vec3, direction: REVec3) -> Self {
        Self {
            origin,
            direction,
            direction_reciprocals: direction.0.reciprocal(),
        }
    }
}

#[debug_requires(is_normalised(wo.0))]
#[debug_requires(is_normalised(normal.0))]
#[debug_ensures(is_normalised(ret.0))]
pub fn reflect(wo: REVec3, normal: REVec3) -> REVec3 { REVec3(-wo.0 + normal.0 * 2. * (normal.0.dot(wo.0))) }

fn schlick(cosθ: Float, η1: Float, η2: Float) -> Float {
    let r0 = ((η1 - η2) / (η1 + η2)).powi(2);
    let r = r0 + (1. - r0) * (1. - cosθ).powi(5);

    r
}

#[debug_requires(is_normalised(wo.0))]
#[debug_requires(is_normalised(normal.0))]
#[debug_ensures(if let Some((vec, _)) = ret { is_normalised(vec.0) } else { true })]
pub fn refract(wo: REVec3, normal: REVec3, incident_index: Float, transmittant_index: Float) -> Option<(REVec3, Float)> {
    let l = -wo.0;

    let index_ratio = incident_index / transmittant_index;

    let cosθ_i = -l.dot(normal.0);
    let sin2θ_i = 1. - cosθ_i * cosθ_i;
    let sin2θ_t = index_ratio * index_ratio * sin2θ_i;

    if sin2θ_t >= 1. {
        return None;
    }

    let cosθ_t = (1. - sin2θ_t).sqrt();

    let refracted_direction = l * index_ratio + normal.0 * (index_ratio * cosθ_i - cosθ_t);

    return Some((REVec3(refracted_direction), schlick(cosθ_i, incident_index, transmittant_index)));
}
