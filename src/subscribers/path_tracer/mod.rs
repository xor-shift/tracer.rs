mod denoiser;
mod double_bufferer;
mod geometry;
mod gpu_tracer;
mod noise_texture;
mod rasteriser;
mod state;
mod texture_set;
mod thing;
mod visualiser;

use wgpu::util::DeviceExt;
use winit::keyboard::SmolStr;

use crate::subscriber::*;
use crate::Application;

use denoiser::Denoiser;
use gpu_tracer::GPUTracer;
use rasteriser::Rasteriser;
use texture_set::TextureSet;
use visualiser::Visualiser;

pub struct PathTracer {
    state: (state::State, wgpu::Buffer),
    old_state: (state::RawState, wgpu::Buffer),

    rasteriser: Rasteriser,
    gpu_tracer: GPUTracer,
    denoiser: Denoiser,
    visualiser: Visualiser,
    visualisation_mode: i32,
}

impl PathTracer {
    fn generate_textures(app: &mut Application, extent: (u32, u32)) -> [TextureSet; 2] { [TextureSet::new(extent, &app.device), TextureSet::new(extent, &app.device)] }

    pub fn new(app: &mut Application) -> color_eyre::Result<PathTracer> {
        let state_buf_desc = wgpu::BufferDescriptor {
            label: Some("tracer.rs state buffer"),
            size: std::mem::size_of::<state::RawState>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        };

        let mut state = state::State::new(app.window.inner_size().into());
        let raw_state = state.frame_start(app);
        let state_buffer = app.device.create_buffer(&state_buf_desc);
        let old_state_buffer = app.device.create_buffer(&state_buf_desc);

        let rasteriser = Rasteriser::new(app, &state_buffer, &old_state_buffer)?;
        let gpu_tracer = GPUTracer::new(app, &rasteriser.texture_set, &state_buffer, &old_state_buffer)?;
        let denoiser = Denoiser::new(app, &rasteriser.texture_set)?;
        let visualiser = Visualiser::new(app, &rasteriser.texture_set)?;

        let this = Self {
            state: (state, state_buffer),
            old_state: (raw_state, old_state_buffer),

            visualiser,
            gpu_tracer,
            denoiser,
            rasteriser,

            visualisation_mode: 0,
        };

        Ok(this)
    }
}

impl Subscriber for PathTracer {
    fn handle_event<'a>(&mut self, app: &'a mut Application, event: &winit::event::Event<()>) -> EventHandlingResult {
        match event {
            winit::event::Event::WindowEvent { window_id, event } if *window_id == app.window.id() => match event {
                winit::event::WindowEvent::KeyboardInput { device_id, event, is_synthetic } if event.state == winit::event::ElementState::Pressed => {
                    let handled = if event.logical_key == winit::keyboard::Key::Character(SmolStr::new_inline("q")) {
                        self.visualisation_mode -= 1;
                        log::debug!("vis mode: {}", self.visualisation_mode);
                        true
                    } else if event.logical_key == winit::keyboard::Key::Character(SmolStr::new_inline("e")) {
                        self.visualisation_mode += 1;
                        log::debug!("vis mode: {}", self.visualisation_mode);
                        true
                    } else {
                        false
                    };

                    if handled {
                        EventHandlingResult::Handled
                    } else {
                        EventHandlingResult::NotHandled
                    }
                }
                _ => EventHandlingResult::NotHandled,
            },
            _ => EventHandlingResult::NotHandled,
        }
    }

    fn resize(&mut self, app: &mut Application, new_size: winit::dpi::PhysicalSize<u32>) {
        self.state.0.resize(new_size);
        self.rasteriser.resize(app, new_size);
        self.gpu_tracer.resize(app, new_size, &self.rasteriser.texture_set);
        self.denoiser.resize(app, &self.rasteriser.texture_set);
        self.visualiser.resize(app, new_size, &self.rasteriser.texture_set);
    }

    fn render(&mut self, app: &mut Application, view: &wgpu::TextureView, encoder: &mut wgpu::CommandEncoder, delta_time: std::time::Duration) {
        let raw_state = self.state.0.frame_start(app);
        app.queue.write_buffer(&self.state.1, 0, bytemuck::bytes_of(&raw_state));
        app.queue.write_buffer(&self.old_state.1, 0, bytemuck::bytes_of(&self.old_state.0));

        self.rasteriser.render(app, delta_time, &self.state.0);
        self.gpu_tracer.render(app, &self.state.0);
        self.denoiser.render(app, denoiser::DenoiserMode::IllumToD0, 1, &self.state.0);
        self.denoiser.render(app, denoiser::DenoiserMode::D0ToD1, 2, &self.state.0);
        self.denoiser.render(app, denoiser::DenoiserMode::D1ToD0, 4, &self.state.0);
        //self.denoiser.render(app, denoiser::DenoiserMode::D0ToD1, 8);
        //self.denoiser.render(app, denoiser::DenoiserMode::D1ToD0, 16);
        self.visualiser.render(app, view, encoder, self.visualisation_mode, &self.state.0);

        self.state.0.frame_end();
        self.old_state.0 = raw_state;
    }
}
