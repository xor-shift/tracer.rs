pub enum Material {
    Air,
    Diffuse { color: [f64; 3] },
    Light { color: [f64; 3] },
    PerfectMirror { color: [f64; 3] },
    PlainGlass { color: [f64; 3] },
    Dielectric { color: [f64; 3], refractive_index: f64 },
    Glossy { color: [f64; 3], glossiness: f64 },
}

pub struct Chunk {
    min: cgmath::Point3<f32>,
    max: cgmath::Point3<f32>,
    depth: usize,
    data: Vec<Material>,
}
