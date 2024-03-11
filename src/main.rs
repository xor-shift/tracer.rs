use winit::event::WindowEvent;

mod essential_stuff;
mod imgui_stuff;

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
    /*let shader_source = include_str!("./imgui_stuff/shader.wgsl");
    let parsed = wgpu::naga::front::wgsl::parse_str(shader_source)?;
    let front_end = wgpu::naga::front::wgsl::Frontend::new();
    println!("{parsed:?}");*/

    let mut stuff = essential_stuff::EssentialStuff::new().await?;

    let mut imgui_stuff = imgui_stuff::Stuff::new(&stuff.device, &stuff.queue)?;

    let event_loop = stuff.event_loop.unwrap();
    stuff.event_loop = None;

    let mut frame_no = 0usize;
    event_loop.run(move |event, window_target| {
        imgui_stuff.event(&event);

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
                    //log::debug!("Rendering frame {}", frame_no);
                    frame_no += 1;

                    let output = stuff.surface.get_current_texture().unwrap();
                    let view = output.texture.create_view(&Default::default());
                    let mut encoder = stuff.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });

                    let ui = imgui_stuff.frame();

                    ui.show_demo_window(&mut true);
                    ui.show_metrics_window(&mut true);

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

                    imgui_stuff.render(&stuff.device, &stuff.queue, &view);

                    output.present();
                }
                _ => {}
            },
            winit::event::Event::NewEvents(_) => {
                //
            }
            winit::event::Event::AboutToWait => {
                stuff.window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}
