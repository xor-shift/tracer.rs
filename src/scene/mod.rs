use crate::*;
use types::*;
use shape::*;

pub struct Scene<GenericShape: Shape> {
    shapes: Vec<GenericShape>,
    
}
