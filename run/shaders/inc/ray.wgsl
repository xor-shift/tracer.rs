struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Intersection {
    distance: f32,
    pos: vec3<f32>,
    normal: vec3<f32>,
    wo: vec3<f32>,
}
