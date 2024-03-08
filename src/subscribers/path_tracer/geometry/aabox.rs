use super::Vertex;

#[repr(C)]
pub struct Box {
    pub min: [f32; 3],
    pub material: u32,
    pub max: [f32; 3],
    padding: u32,
}

impl Box {
    pub const fn new(min: [f32; 3], max: [f32; 3], material: u32) -> Box {
        Self {
            min, //
            max,
            material,
            padding: 0,
        }
    }

    pub fn into_vertices(&self) -> [Vertex; 3 * 6 * 2] {
        todo!();
    }
}
