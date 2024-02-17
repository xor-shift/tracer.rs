use cgmath::InnerSpace;

use stuff::rng::{RandomNumberEngine, UniformRandomBitGenerator};

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct RawMainUniform {
    pub dimensions: [u32; 2],
    pub frame_no: u32,
    pub current_instant: f32,
    pub seed: [u32; 4],
    pub visualisation_mode: i32,
    pub padding_0: [u8; 12],
    pub camera_position: [f32; 3],
    pub padding_1: [u8; 4],
}

#[derive(Copy, Clone)]
pub struct UniformGenerator {
    pub dimensions: (u32, u32),
    pub frame_no: u32,
    pub started_at: std::time::Instant,
    pub last_render_at: std::time::Instant,
    pub generator: stuff::rng::engines::Xoshiro256PP,
    pub next_seed: [u32; 4],
    pub visualisation_mode: i32,

    pub camera_position: [f32; 3],
    pub camera_rotation: [f32; 3],
    pub pending_movement: [f32; 3],
    pub pending_rotation: [f32; 3],
}

impl UniformGenerator {
    pub fn new(dimensions: (u32, u32)) -> Self {
        let mut rd = stuff::rng::engines::RandomDevice::new();
        let mut gen = stuff::rng::engines::Xoshiro256PP::new();
        gen.seed_from_result(rd.generate());

        UniformGenerator {
            dimensions,
            frame_no: 0,
            started_at: std::time::Instant::now(),
            last_render_at: std::time::Instant::now(),
            generator: gen,
            next_seed: [0u32; 4],
            visualisation_mode: 0,
            camera_position: [0., 0., 0.],
            camera_rotation: [0., 0., 0.],
            pending_movement: [0., 0., 0.],
            pending_rotation: [0., 0., 0.],
        }
    }

    pub fn generate(&mut self) -> RawMainUniform {
        let now = std::time::Instant::now();
        let update_delta = (now - self.last_render_at).as_secs_f32();
        let speed = 5.;

        let movement = cgmath::Vector3::<f32>::from(self.pending_movement) * update_delta * speed;
        self.camera_position[0] += movement.x;
        self.camera_position[1] += movement.y;
        self.camera_position[2] += movement.z;

        self.pending_movement = [0., 0., 0.];
        self.pending_rotation = [0., 0., 0.];
        self.last_render_at = now;

        RawMainUniform {
            dimensions: self.dimensions.into(),
            frame_no: self.frame_no,
            current_instant: (now - self.started_at).as_secs_f32(),
            seed: self.next_seed,
            visualisation_mode: self.visualisation_mode,
            padding_0: [0; 12],
            camera_position: self.camera_position,
            padding_1: [0; 4],
        }
    }

    pub fn frame_start(&mut self) -> RawMainUniform { self.generate() }

    pub fn frame_end(&mut self) {
        self.frame_no += 1;
        self.next_seed = [self.generator.generate() as u32, self.generator.generate() as u32, self.generator.generate() as u32, self.generator.generate() as u32];
    }

    pub fn reset(&mut self) {
        self.frame_no = 0;
        self.started_at = std::time::Instant::now();
    }
}
