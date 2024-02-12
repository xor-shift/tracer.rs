use crate::material::EMaterial;
use crate::types::*;
use crate::*;

#[derive(Clone)]
pub struct Intersection {
    pub position: Vec3,
    pub distance: Float,
    pub wo: REVec3,

    pub normal: REVec3,

    pub texture_coords: VecUV,

    pub dp_du: REVec3,
    pub dp_dv: REVec3,
    pub reflection_to_surface: Mat3x3,

    pub material_id: u16,
}

impl Intersection {
    pub fn select_best(lhs: Option<Intersection>, rhs: Option<Intersection>) -> Option<Intersection> {
        match (lhs, rhs) {
            (None, None) => None,
            (Some(v), None) => Some(v),
            (None, Some(v)) => Some(v),
            (Some(l), Some(r)) => Some(if l.distance < r.distance { l } else { r }),
        }
    }

    pub fn going_in(&self) -> bool { 0. < self.wo.0.dot(self.normal.0) }

    /// The normal point in the direction of `self.wo`.
    pub fn oriented_normal(&self) -> REVec3 { REVec3(if self.going_in() { self.normal.0 } else { -self.normal.0 }) }

    /// Converts a reflection-space vector to a surface-space vector.
    pub fn convert_to_surface_space(&self, vec: REVec3) -> SVec3 { SVec3(self.reflection_to_surface * vec.0) }

    /// Converts a surface-space vector to a reflection-space vector. Use when converting from importance-sampled samples.
    pub fn convert_to_reflection_space(&self, vec: SVec3) -> REVec3 { REVec3(self.reflection_to_surface.transpose() * vec.0) }
}
