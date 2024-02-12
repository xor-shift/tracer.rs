use crate::subscriber::Subscriber;

//use egui_wgpu_backend::{RenderPass, ScreenDescriptor};
// use egui_winit::{Platform, PlatformDescriptor};

pub struct GUISubscriber {}

impl GUISubscriber {
    pub fn new() -> GUISubscriber {
        Self {}
    }
}

impl Subscriber for GUISubscriber {

}
