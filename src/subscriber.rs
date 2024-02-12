use crate::Application;

pub enum EventHandlingResult {
    ConsumedCF(winit::event_loop::ControlFlow),
    Consumed,
    Handled,
    NotHandled,
    RequestExit,
}

pub trait Subscriber {
    fn handle_event<'a>(&mut self, app: &'a mut Application, event: &winit::event::Event<()>) -> EventHandlingResult { EventHandlingResult::NotHandled }

    fn update(&mut self, app: &mut Application, delta_time: std::time::Duration) {}

    fn render(&mut self, app: &mut Application, view: &wgpu::TextureView, encoder: &mut wgpu::CommandEncoder, delta_time: std::time::Duration) {}

    fn resize(&mut self, app: &mut Application, new_size: winit::dpi::PhysicalSize<u32>) {}
}
