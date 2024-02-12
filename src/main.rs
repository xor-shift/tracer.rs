// #![feature(adt_const_params)]
#![feature(auto_traits)]
#![feature(const_trait_impl)]
#![feature(generic_const_exprs)]
#![feature(let_chains)]
#![feature(macro_metavar_expr)]
#![feature(negative_impls)]
#![feature(new_uninit)]
#![feature(stmt_expr_attributes)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(incomplete_features)]
#![allow(mixed_script_confusables)]

mod subscribers;
mod app;
mod input_store;
pub mod subscriber;

pub use app::Application;

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

    log::debug!("Hello, world!");

    let mut app = Application::new().await.expect("failed to create the app");

    let subscriber = Box::new(subscribers::fps_tracker::FPSTracker::new());
    app.add_subscriber(subscriber);

    let subscriber = Box::new(subscribers::gui::GUISubscriber::new());
    app.add_subscriber(subscriber);

    let subscriber = Box::new(subscribers::path_tracer::PathTracer::new(&mut app));
    app.add_subscriber(subscriber);

    app.run().await.expect("app exited with an error");
}
