use crate::*;
use types::*;
use shape::*;

pub struct Scene<GenericShape: Shape> {
    pub skybox: Skybox,
    pub shapes: Vec<GenericShape>,
}
