mod aabox;
mod triangle;
mod vertex;

pub use aabox::Box;
pub use triangle::triangles_into_vertices;
pub use triangle::Triangle;
pub use vertex::Vertex;

pub(self) const fn vec3_to_pt(v: [f32; 3]) -> [f32; 4] { [v[0], v[1], v[2], 1.] }

pub(self) const fn vec3_to_vec(v: [f32; 3]) -> [f32; 4] { [v[0], v[1], v[2], 0.] }
