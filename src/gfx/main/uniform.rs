use stuff::rng::{RandomNumberEngine, UniformRandomBitGenerator};

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct RawMainUniform {
    pub frame_no: u32,
    pub current_instant: f32,
    pub seed: [u32; 4],
}

#[derive(Copy, Clone)]
pub struct UniformGenerator {
    pub frame_no: u32,
    pub started_at: std::time::Instant,
    pub generator: stuff::rng::engines::Xoshiro256PP,
    pub next_seed: [u32; 4],
}

impl UniformGenerator {
    pub fn new() -> Self {
        let mut rd = stuff::rng::engines::RandomDevice::new();
        let mut gen = stuff::rng::engines::Xoshiro256PP::new();
        gen.seed_from_result(rd.generate());

        UniformGenerator {
            frame_no: 0,
            started_at: std::time::Instant::now(),
            generator: gen,
            next_seed: [0u32; 4],
        }
    }

    pub fn generate(&self) -> RawMainUniform {
        RawMainUniform {
            frame_no: self.frame_no,
            current_instant: (std::time::Instant::now() - self.started_at).as_secs_f32(),
            seed: self.next_seed,
        }
    }

    pub fn frame_start(&self) -> RawMainUniform { self.generate() }

    pub fn frame_end(&mut self) {
        self.frame_no += 1;
        self.next_seed = [self.generator.generate() as u32, self.generator.generate() as u32, self.generator.generate() as u32, self.generator.generate() as u32];
    }

    pub fn reset(&mut self) {
        self.frame_no = 0;
        self.started_at = std::time::Instant::now();
    }
}
