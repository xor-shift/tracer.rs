mod geometry;
mod gpu_tracer;
mod noise_texture;
mod rasteriser;
mod texture_set;
mod uniform;
mod vertex;
mod visualiser;

use wgpu::util::DeviceExt;
use winit::keyboard::SmolStr;

use crate::subscriber::*;
use crate::Application;

use gpu_tracer::GPUTracer;
use rasteriser::Rasteriser;
use texture_set::TextureSet;
use visualiser::Visualiser;

pub struct PathTracer {
    tex_dims: (u32, u32),

    gpu_tracer: GPUTracer,
    visualiser: Visualiser,
    rasteriser: Rasteriser,
    visualisation_mode: i32,
}

impl PathTracer {
    fn generate_textures(app: &mut Application, extent: (u32, u32)) -> [TextureSet; 2] { [TextureSet::new(extent, &app.device), TextureSet::new(extent, &app.device)] }

    pub fn new(app: &mut Application) -> color_eyre::Result<PathTracer> {
        let rasteriser = Rasteriser::new(app)?;
        let gpu_tracer = GPUTracer::new(app, &rasteriser.texture_set, &rasteriser.geometry_buffer)?;
        let visualiser = Visualiser::new(app, &rasteriser.texture_set, &rasteriser.geometry_buffer)?;

        let this = Self {
            tex_dims: app.window.inner_size().into(),

            visualiser,
            gpu_tracer,
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
        self.tex_dims = (new_size.width, new_size.height);

        self.rasteriser.resize(app, new_size);
        self.gpu_tracer.resize(app, new_size, &self.rasteriser.texture_set, &self.rasteriser.geometry_buffer);
        self.visualiser.resize(app, new_size, &self.rasteriser.texture_set, &self.rasteriser.geometry_buffer);
    }

    fn render(&mut self, app: &mut Application, view: &wgpu::TextureView, encoder: &mut wgpu::CommandEncoder, delta_time: std::time::Duration) {
        if app.input_store.is_pressed(winit::keyboard::Key::Character("w".into())) {
            self.gpu_tracer.uniform_generator.pending_movement[2] += 1.;
        }
        if app.input_store.is_pressed(winit::keyboard::Key::Character("a".into())) {
            self.gpu_tracer.uniform_generator.pending_movement[0] -= 1.;
        }
        if app.input_store.is_pressed(winit::keyboard::Key::Character("s".into())) {
            self.gpu_tracer.uniform_generator.pending_movement[2] -= 1.;
        }
        if app.input_store.is_pressed(winit::keyboard::Key::Character("d".into())) {
            self.gpu_tracer.uniform_generator.pending_movement[0] += 1.;
        }
        if app.input_store.is_pressed(winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space)) {
            self.gpu_tracer.uniform_generator.pending_movement[1] += 1.;
        }
        if app.input_store.is_pressed(winit::keyboard::Key::Named(winit::keyboard::NamedKey::Shift)) {
            self.gpu_tracer.uniform_generator.pending_movement[1] -= 1.;
        }

        self.rasteriser.render(app, delta_time);
        self.gpu_tracer.render(app, self.rasteriser.uniform.camera_position.into());
        self.visualiser.render(app, view, encoder, self.visualisation_mode);
    }
}
