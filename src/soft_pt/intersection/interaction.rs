use crate::*;
use types::*;

#[derive(Clone)]
pub struct Interaction {
    pub normal: REVec3,
    pub wi: REVec3,
    pub wo: REVec3,

    /// brdf * cosθ / pdf(ωi)
    pub weight: Float,

    pub attenuation: color::LinearRGB,
    pub emittance: color::LinearRGB,
}

impl Interaction {
    
}
