#![allow(incomplete_features)]
#![feature(allocator_api)]
#![feature(ascii_char)]
#![feature(const_mut_refs)]
#![feature(exclusive_range_pattern)]
#![feature(generic_const_exprs)]
#![feature(iter_partition_in_place)]
#![feature(num_midpoint)]
#![feature(stmt_expr_attributes)]

use winit::event::WindowEvent;

mod basic_octree;
mod essential_stuff;
mod imgui_stuff;
mod input_tracker;
mod scene;
mod state;
mod tracer;

#[tokio::main]
async fn main() {
    color_backtrace::install();

    if let Err(v) = dotenv::dotenv() {
        env_logger::init();
        log::warn!("failed to initialise dotenv: {}", v);
    } else {
        env_logger::init();
    }

    log::debug!("Hello, world!");

    foo().await.unwrap();
}

async fn foo() -> color_eyre::Result<()> {
    let mut stuff = essential_stuff::EssentialStuff::new().await?;
    let mut imgui_stuff = imgui_stuff::Stuff::new(&stuff.device, &stuff.queue)?;
    let mut input_tracker = input_tracker::InputTracker::new();
    //let mut state = tracer::state::State::new();
    let mut state = state::State::new();
    let mut tracer = tracer::Tracer::new(&stuff.device, &stuff.queue)?;

    let event_loop = stuff.event_loop.unwrap();
    stuff.event_loop = None;

    let mut frame_no = 0usize;
    event_loop.run(move |event, window_target| {
        imgui_stuff.event(&event);

        if let winit::event::Event::WindowEvent { window_id: _, event } = &event {
            input_tracker.window_event(&event);
            state.window_event(&event);
            tracer.window_event(&stuff.device, &stuff.queue, &event);
        }

        match event {
            winit::event::Event::WindowEvent { window_id, event } if window_id == stuff.window.id() => match event {
                WindowEvent::Resized(physical_size) => {
                    stuff.surface_config.width = physical_size.width;
                    stuff.surface_config.height = physical_size.height;
                    stuff.surface.configure(&stuff.device, &stuff.surface_config);
                    stuff.window.request_redraw();
                }
                WindowEvent::CloseRequested => {
                    window_target.exit();
                }
                WindowEvent::RedrawRequested => {
                    let output = stuff.surface.get_current_texture().unwrap();
                    let view = output.texture.create_view(&Default::default());
                    let mut encoder = stuff.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });

                    let ui = imgui_stuff.frame();

                    input_tracker.frame_process_events(ui.io().want_capture_mouse, ui.io().want_capture_keyboard);

                    //ui.show_demo_window(&mut true);
                    ui.show_metrics_window(&mut true);

                    ui.window("settings").build(|| {
                        let mut as_usize = state.visualisation_mode as usize;

                        use state::VisualisationMode::*;
                        let list = [PathTrace, Denoise0, Denoise1, PathTraceAlbedo, Denoise0Albedo, Denoise1Albedo, Normal, AbsNormal, DistFromOrigin];

                        ui.combo("visualisation mode", &mut as_usize, &list, |v| v.as_str().into());
                        ui.text(format!("pos: [{:.2}, {:.2}, {:.2}]", state.position.x, state.position.y, state.position.z));
                        ui.text(format!("yaw: {:.3}, pitch: {:.3}, roll: {:.3}", state.rotation[0], state.rotation[1], state.rotation[2]));

                        state.visualisation_mode = list[as_usize];

                        ui.slider("FOV", 15., 179., &mut state.fov.0);

                        ui.text("mouse acceleration");
                        ui.checkbox("enabled", &mut state.mouse_accel_enabled);
                        if state.mouse_accel_enabled {
                            ui.input_scalar("param", &mut state.mouse_accel_param).step(0.1).build()
                        } else {
                            ui.input_scalar("speed", &mut state.linear_mouse_speed).step(1.).build()
                        };
                    });

                    {
                        let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("clear pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 0. }),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            occlusion_query_set: None,
                            timestamp_writes: None,
                        });
                    }

                    stuff.device.poll(wgpu::Maintain::WaitForSubmissionIndex(stuff.queue.submit(std::iter::once(encoder.finish()))));

                    let raw_state = state.pre_render(&input_tracker);
                    tracer.render(&stuff.device, &stuff.queue, &view, &raw_state);

                    imgui_stuff.render(&stuff.device, &stuff.queue, &view);

                    output.present();
                }
                _ => {}
            },
            winit::event::Event::NewEvents(_) => {
                input_tracker.frame_start();
            }
            winit::event::Event::AboutToWait => {
                input_tracker.frame_end();
                stuff.window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}
