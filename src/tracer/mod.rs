use crate::state::RawState;

mod texture_set;

mod denoiser;
mod path_tracer;
mod visualiser;

pub struct Tracer {
    texture_set: texture_set::TextureSet,
    path_tracer: path_tracer::PathTracer,
    visualiser: visualiser::Visualiser,
}

impl Tracer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> color_eyre::Result<Tracer> {
        let texture_set = texture_set::TextureSet::new(device, (1, 1));
        Ok(Self {
            texture_set,
            path_tracer: path_tracer::PathTracer::new(device, queue)?,
            visualiser: visualiser::Visualiser::new(device, queue)?,
        })
    }

    pub fn window_event(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, event: &winit::event::WindowEvent) {
        if let winit::event::WindowEvent::Resized(new_size) = event {
            let dimensions = (*new_size).into();

            self.texture_set = texture_set::TextureSet::new(device, dimensions);

            self.path_tracer.resize(device, queue, dimensions, &self.texture_set);
            self.visualiser.resize(device, queue, dimensions, &self.texture_set);
        }
    }

    pub fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, unto: &wgpu::TextureView, state: &RawState) {
        self.path_tracer.render(device, queue, &self.texture_set, state, (8, 8));
        self.visualiser.render(device, queue, unto, state);
    }
}
