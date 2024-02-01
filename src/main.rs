// #![feature(adt_const_params)]
#![feature(auto_traits)]
#![feature(const_trait_impl)]
#![feature(generic_const_exprs)]
#![feature(let_chains)]
#![feature(macro_metavar_expr)]
#![feature(negative_impls)]
#![feature(new_uninit)]
#![feature(stmt_expr_attributes)]
#![allow(incomplete_features)]
#![allow(mixed_script_confusables)]

use std::sync::Arc;
use std::sync::Mutex;

mod camera;
mod color;
mod gfx;
mod intersection;
mod light;
mod material;
mod ray;
mod renderer;
mod scene;
mod shape;
mod types;

pub use color::*;
pub use intersection::*;
pub use light::*;
pub use material::*;
pub use ray::*;
pub use renderer::*;
pub use scene::*;
pub use shape::*;
pub use types::*;

mod fun;

pub use log::{debug, error, info, log_enabled, Level};
use std::default::Default;
use wgpu::util::DeviceExt;
use winit::event::WindowEvent;

use cgmath::prelude::*;

#[tokio::main]
pub async fn main() {
    color_backtrace::install();

    if let Err(v) = dotenv::dotenv() {
        env_logger::init();
        log::warn!("failed to initialise dotenv: {}", v);
    } else {
        env_logger::init();
    }

    debug!("Hello, world!");

    let mut app = gfx::Application::new().await.expect("failed to create the app");

    let subscriber = gfx::main::ComputeTest::new(&mut app);
    app.add_subscriber(subscriber);

    app.run().await.expect("app exited with an error");
}
