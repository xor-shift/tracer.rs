use wgpu::Instance;
use winit::event_loop;
use winit::event_loop::EventLoop;
use winit::window::Window;
use winit::window::WindowBuilder;

pub trait Subscriber {
    fn handle_event<'a>(&mut self, app: &'a mut Application, event: &winit::event::Event<'a, ()>) -> EventHandlingResult;

    fn update(&mut self, app: &mut Application, delta_time: std::time::Duration) {}

    fn render(&mut self, app: &mut Application, view: &wgpu::TextureView, encoder: &mut wgpu::CommandEncoder, delta_time: std::time::Duration) {}
}

pub struct Application {
    event_loop: Option<EventLoop<()>>,

    pub surface: wgpu::Surface,
    pub window: Window,

    pub instance: Instance,

    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    // pub queue_graphics: wgpu::Queue,
    // pub queue_compute: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,

    pub last_update: std::time::Instant,
    pub last_sim_update: std::time::Instant,
    pub last_render: std::time::Instant,

    subscribers: Option<Vec<Box<dyn Subscriber>>>,
}

pub enum EventHandlingResult {
    ConsumedCF(winit::event_loop::ControlFlow),
    Consumed,
    Handled,
    NotHandled,
}

impl Application {
    pub async fn new() -> color_eyre::Result<Self> {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new().build(&event_loop).unwrap();
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let surface = unsafe { instance.create_surface(&window) }.unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter().copied().filter(|f| f.is_srgb()).next().unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let mut ret = Self {
            event_loop: Some(event_loop),

            surface: surface,
            window,

            instance,

            device,
            queue,
            config,
            size,

            last_update: std::time::Instant::now(),
            last_sim_update: std::time::Instant::now(),
            last_render: std::time::Instant::now(),

            subscribers: Some(vec![Box::new(CoreSubscriber {})]),
        };

        Ok(ret)
    }

    pub fn add_subscriber(&mut self, subscriber: Box<dyn Subscriber>) {
        if let Some(subscribers) = &mut self.subscribers {
            subscribers.push(subscriber);
        } else {
            self.subscribers = Some(vec![subscriber]);
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn subscriber_guard<Ret: Sized, Fun: FnMut(&mut Self, &mut Vec<Box<dyn Subscriber>>) -> Ret>(&mut self, mut fun: Fun) -> Ret {
        let mut subscribers = std::mem::replace(&mut self.subscribers, None).unwrap_or(Vec::new());

        let ret = fun(self, &mut subscribers);

        if let Some(mut appended_subscribers) = std::mem::replace(&mut self.subscribers, None) {
            subscribers.append(&mut appended_subscribers);
        }
        self.subscribers = Some(subscribers);

        ret
    }

    fn apply_to_subscribers<Fun: FnMut(&mut Self, &mut dyn Subscriber) -> ()>(&mut self, mut fun: Fun) {
        self.subscriber_guard(|this, subscribers| {
            for subscriber in subscribers {
                fun(this, subscriber.as_mut());
            }
        })
    }

    fn do_update(&mut self) {
        let now = std::time::Instant::now();
        let delta = now - self.last_update;
        self.last_update = now;

        self.apply_to_subscribers(|this: &mut Self, subscriber: &mut dyn Subscriber| subscriber.update(this, delta));
    }

    fn do_render(&mut self) -> color_eyre::Result<(), wgpu::SurfaceError> {
        let now = std::time::Instant::now();
        let delta = now - self.last_render;
        self.last_render = now;

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });

        self.apply_to_subscribers(|this: &mut Self, subscriber: &mut dyn Subscriber| subscriber.render(this, &view, &mut encoder, delta));

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn try_handle_event_internally(&mut self, event: &winit::event::Event<'_, ()>) -> EventHandlingResult {
        match event {
            winit::event::Event::RedrawRequested(window_id) if window_id == &self.window.id() => {
                self.do_update();
                match self.do_render() {
                    _ => {}
                }

                EventHandlingResult::Consumed
            }
            winit::event::Event::MainEventsCleared => {
                self.window.request_redraw();
                EventHandlingResult::Consumed
            }
            winit::event::Event::WindowEvent { window_id, event } if window_id == &self.window.id() => {
                EventHandlingResult::NotHandled //
            }
            _ => EventHandlingResult::NotHandled,
        }
    }

    #[allow(unreachable_code)]
    pub async fn run(mut self) -> color_eyre::Result<()> {
        let event_loop = self.event_loop;
        let event_loop = event_loop.unwrap();
        self.event_loop = None;

        event_loop.run(move |event, _, control_flow| {
            let res = self.try_handle_event_internally(&event);

            let res = match res {
                EventHandlingResult::ConsumedCF(code) => {
                    *control_flow = code;
                    return;
                }
                EventHandlingResult::Consumed => {
                    return;
                }
                EventHandlingResult::Handled | EventHandlingResult::NotHandled => self.subscriber_guard(|this, subscribers| {
                    let mut ret = EventHandlingResult::NotHandled;

                    for subscriber in subscribers {
                        let temp = subscriber.handle_event(this, &event);

                        ret = match (ret, temp) {
                            (EventHandlingResult::NotHandled, temp) => temp,
                            (ret, EventHandlingResult::NotHandled) => ret,

                            (EventHandlingResult::Handled, EventHandlingResult::Handled) => EventHandlingResult::Handled,
                            (_, EventHandlingResult::Consumed) => EventHandlingResult::Consumed,
                            (_, EventHandlingResult::ConsumedCF(cf)) => EventHandlingResult::ConsumedCF(cf),

                            (EventHandlingResult::Consumed | EventHandlingResult::ConsumedCF(_), _) => unreachable!(),
                        };

                        if !matches!(ret, EventHandlingResult::Handled) && !matches!(ret, EventHandlingResult::NotHandled) {
                            break;
                        }
                    }

                    ret
                }),
            };

            match res {
                EventHandlingResult::ConsumedCF(cf) => *control_flow = cf,
                _ => {}
            };
        });

        std::process::exit(0);
    }
}

struct CoreSubscriber {}

impl CoreSubscriber {}

impl Subscriber for CoreSubscriber {
    fn handle_event(&mut self, app: &mut Application, event: &winit::event::Event<'_, ()>) -> EventHandlingResult {
        match event {
            winit::event::Event::WindowEvent { ref event, window_id } if window_id == &app.window.id() => match event {
                winit::event::WindowEvent::CloseRequested => EventHandlingResult::ConsumedCF(winit::event_loop::ControlFlow::ExitWithCode(0)),
                winit::event::WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    app.resize(**new_inner_size);
                    EventHandlingResult::Handled
                }
                winit::event::WindowEvent::Resized(physical_size) => {
                    app.resize(*physical_size);
                    EventHandlingResult::Handled
                }
                _ => EventHandlingResult::NotHandled,
            },
            _ => EventHandlingResult::NotHandled,
        }
    }

    fn render(&mut self, app: &mut Application, view: &wgpu::TextureView, encoder: &mut wgpu::CommandEncoder, delta_time: std::time::Duration) {
        let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
    }
}