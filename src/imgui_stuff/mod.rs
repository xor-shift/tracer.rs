use std::{collections::HashMap, marker::PhantomData};

mod font;

const fn imgui_vert_desc() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x2,
        1 => Float32x2,
        2 => Uint32,
    ];

    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<imgui::DrawVert>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRIBUTES,
    }
}

fn draw_vert_to_bytes(vert: &imgui::DrawVert) -> [u8; std::mem::size_of::<imgui::DrawVert>()] { unsafe { std::mem::transmute(*vert) } }

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    dimensions: [u32; 2],
    padding: [u8; 8],
}

struct WGPUStuff {
    pipeline: wgpu::RenderPipeline,

    index_buffer_descriptor: wgpu::BufferDescriptor<'static>,
    index_buffer: wgpu::Buffer,
    vertex_buffer_descriptor: wgpu::BufferDescriptor<'static>,
    vertex_buffer: wgpu::Buffer,

    uniform_bg: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,

    texture_bgl: wgpu::BindGroupLayout,
}

impl WGPUStuff {
    const INDEX_SIZE: usize = std::mem::size_of::<u32>();
    const VERTEX_SIZE: usize = std::mem::size_of::<imgui::DrawVert>();

    fn recreate_or_reuse_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer, desc: &wgpu::BufferDescriptor<'static>, desired_size: u64) -> Option<wgpu::Buffer> {
        if buffer.size() >= desired_size {
            return None;
        }

        log::trace!("recreating a buffer to increase its size from {} to {desired_size}", buffer.size());

        let new_desc = wgpu::BufferDescriptor { size: desired_size, ..*desc };
        let buffer = device.create_buffer(&new_desc);

        Some(buffer)
    }

    fn resize_buffers(&mut self, new_idx_count: usize, new_vtx_count: usize, device: &wgpu::Device) {
        if let Some(new_buffer) = Self::recreate_or_reuse_buffer(device, &self.index_buffer, &self.index_buffer_descriptor, (new_idx_count * Self::INDEX_SIZE) as u64) {
            self.index_buffer = new_buffer;
        }

        if let Some(new_buffer) = Self::recreate_or_reuse_buffer(device, &self.vertex_buffer, &self.vertex_buffer_descriptor, (new_vtx_count * Self::VERTEX_SIZE) as u64) {
            self.vertex_buffer = new_buffer;
        }
    }

    fn update_buffers(&self, index_stuff: (usize, &[u16]), vertex_stuff: (usize, &[imgui::DrawVert]), queue: &wgpu::Queue) {
        let ibuf_offset = index_stuff.0 * Self::INDEX_SIZE;
        let ibuf_data = index_stuff.1.iter().map(|&v| [(v & 0xFF) as u8, (v >> 8) as u8, 0u8, 0u8]).flatten().collect::<Vec<_>>();
        queue.write_buffer(&self.index_buffer, ibuf_offset as u64, ibuf_data.as_slice());

        let vbuf_offset = vertex_stuff.0 * Self::VERTEX_SIZE;
        let vbuf_data = vertex_stuff.1.iter().map(draw_vert_to_bytes).flatten().collect::<Vec<_>>();
        queue.write_buffer(&self.vertex_buffer, vbuf_offset as u64, vbuf_data.as_slice());
    }
}

pub struct Stuff {
    imgui_context: imgui::Context,
    font_stuff: font::FontStuff,
    last_frame: std::time::Instant,
    window_size: [u32; 2],

    wgpu_stuff: WGPUStuff,
}

impl Stuff {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> color_eyre::Result<Stuff> {
        let shader_contents = include_str!("./shader.wgsl");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tracer.rs imgui shader"),
            source: wgpu::ShaderSource::Wgsl(shader_contents.into()),
        });

        let vertex_buffer_descriptor = wgpu::BufferDescriptor {
            label: Some("tracer.rs imgui vertex buffer"),
            size: 0,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        let vertex_buffer = device.create_buffer(&vertex_buffer_descriptor);

        let index_buffer_descriptor = wgpu::BufferDescriptor {
            label: Some("tracer.rs imgui index buffer"),
            size: 0,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        let index_buffer = device.create_buffer(&vertex_buffer_descriptor);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tracer.rs imgui uniform buffer"),
            size: std::mem::size_of::<Uniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tracer.rs imgui bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tracer.rs imgui bind group layout"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        let texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tracer.rs imgui texture bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tracer.rs imgui pipeline layout"),
            bind_group_layouts: &[&uniform_bgl, &texture_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tracer.rs imgui pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[imgui_vert_desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let mut imgui_context = imgui::Context::create();
        imgui_context.fonts().add_font(&[imgui::FontSource::DefaultFontData { config: None }]);
        let font_stuff = font::FontStuff::new(&mut imgui_context.fonts(), &texture_bgl, device, queue);

        Ok(Self {
            imgui_context,
            font_stuff,
            last_frame: std::time::Instant::now(),
            window_size: [512, 512],

            wgpu_stuff: WGPUStuff {
                pipeline,

                index_buffer_descriptor,
                index_buffer,
                vertex_buffer_descriptor,
                vertex_buffer,

                uniform_bg,
                uniform_buffer,

                texture_bgl,
            },
        })
    }

    pub fn event<T: Sized>(&mut self, event: &winit::event::Event<T>) {
        match event {
            winit::event::Event::WindowEvent { window_id, event } => match event {
                winit::event::WindowEvent::Resized(physical_size) => {
                    self.window_size = (*physical_size).into();
                    self.imgui_context.io_mut().display_size = (*physical_size).into();
                }
                winit::event::WindowEvent::CursorMoved { device_id, position } => {
                    self.imgui_context.io_mut().add_mouse_pos_event((*position).into());
                }
                winit::event::WindowEvent::MouseInput { device_id, state, button } => {
                    let im_button = match *button {
                        winit::event::MouseButton::Left => Some(imgui::MouseButton::Left),
                        winit::event::MouseButton::Right => Some(imgui::MouseButton::Right),
                        winit::event::MouseButton::Middle => Some(imgui::MouseButton::Middle),
                        winit::event::MouseButton::Back => Some(imgui::MouseButton::Extra1),
                        winit::event::MouseButton::Forward => Some(imgui::MouseButton::Extra2),
                        _ => None,
                    };

                    if let Some(button) = im_button {
                        self.imgui_context.io_mut().add_mouse_button_event(button, state.is_pressed());
                    }
                }
                _ => {}
            },
            winit::event::Event::NewEvents(_) => {
                let now = std::time::Instant::now();
                let elapsed = now - self.last_frame;
                self.last_frame = now;

                self.imgui_context.io_mut().update_delta_time(elapsed);
            }
            _ => {}
        }
    }

    pub fn frame(&mut self) -> &mut imgui::Ui { self.imgui_context.new_frame() }

    // what a mess
    pub fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, output: &wgpu::TextureView) {
        let uniforms = Uniforms {
            dimensions: self.window_size,
            padding: [0; 8],
        };
        queue.write_buffer(&self.wgpu_stuff.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let draw_data = self.imgui_context.render();
        log::trace!("a total of {} indices referring to {} vertices will be rendered", draw_data.total_idx_count, draw_data.total_vtx_count);

        // i think that the pointer containing draw lists can be nullptr when this is the case which
        // causes an issue with slice::from_raw_parts
        if draw_data.total_idx_count == 0 {
            return;
        }

        self.wgpu_stuff.resize_buffers(draw_data.total_idx_count as usize, draw_data.total_vtx_count as usize, device);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tracer.rs imgui command encoder"),
        });

        // offsets to buffer indices across draw lists
        let mut index_buffer_offset = 0usize;
        let mut vertex_buffer_offset = 0usize;

        for (list_no, list) in draw_data.draw_lists().enumerate() {
            let index_list = list.idx_buffer();
            let vertex_list = list.vtx_buffer();
            log::trace!("draw list #{list_no} has {} indices and {} vertices", index_list.len(), vertex_list.len());

            self.wgpu_stuff.update_buffers((index_buffer_offset, index_list), (vertex_buffer_offset, vertex_list), queue);

            let commands = list.commands();

            for (command_no, command) in commands.enumerate() {
                match command {
                    imgui::DrawCmd::Elements { count, cmd_params } => {
                        log::trace!("command #{command_no} is an imgui::DrawCmd::Elements {{ {count}, .. }}");
                        log::trace!(
                            "params: clip = {:?}, tid = {}, voff = {}, ioff = {}",
                            cmd_params.clip_rect,
                            cmd_params.texture_id.id(),
                            cmd_params.vtx_offset,
                            cmd_params.idx_offset
                        );

                        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("tracer.rs rasteriser pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: output,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            occlusion_query_set: None,
                            timestamp_writes: None,
                        });

                        render_pass.set_pipeline(&self.wgpu_stuff.pipeline);
                        render_pass.set_bind_group(0, &self.wgpu_stuff.uniform_bg, &[]);
                        render_pass.set_bind_group(1, &self.font_stuff.bind_group, &[]);
                        render_pass.set_vertex_buffer(0, self.wgpu_stuff.vertex_buffer.slice(..));
                        render_pass.set_index_buffer(self.wgpu_stuff.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.set_scissor_rect(
                            cmd_params.clip_rect[0] as u32,
                            cmd_params.clip_rect[1] as u32,
                            (cmd_params.clip_rect[2] - cmd_params.clip_rect[0]).round() as u32,
                            (cmd_params.clip_rect[3] - cmd_params.clip_rect[1]).round() as u32,
                        );

                        let index_buffer_start = (index_buffer_offset + cmd_params.idx_offset) as u32;
                        render_pass.draw_indexed(
                            index_buffer_start..index_buffer_start + count as u32, //
                            vertex_buffer_offset as i32,
                            0..1,
                        );
                    }
                    imgui::DrawCmd::ResetRenderState => {
                        log::trace!("command #{command_no} is an imgui::DrawCmd::ResetRenderState");
                        todo!();
                    }
                    imgui::DrawCmd::RawCallback { callback, raw_cmd } => {
                        log::trace!("command #{command_no} imgui::DrawCmd::RawCallback {{ {callback:?}, {raw_cmd:?} }}");
                        todo!();
                    }
                }
            }

            index_buffer_offset += index_list.len();
            vertex_buffer_offset += vertex_list.len();
        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}

struct PretendBorrower<'a, T: ?Sized> {
    previous: PhantomData<T>,
    current: PhantomData<&'a ()>,
}

pub struct Frame<BorrowedStuff: ?Sized = ()> {
    borrows: PhantomData<BorrowedStuff>,
    texture_ptrs: HashMap<wgpu::Id<wgpu::Texture>, *const wgpu::Texture>,
}

impl<BorrowedStuff: ?Sized> Frame<BorrowedStuff> {
    fn new() -> Frame<BorrowedStuff> {
        Self {
            borrows: PhantomData {},
            texture_ptrs: HashMap::new(),
        }
    }

    pub fn borrow_another<'t>(self, texture: &'t wgpu::Texture) -> Frame<PretendBorrower<'t, BorrowedStuff>> {
        let mut texture_ptrs = self.texture_ptrs;
        texture_ptrs.insert(texture.global_id(), texture as *const wgpu::Texture);

        Frame::<PretendBorrower<'t, BorrowedStuff>> { borrows: PhantomData {}, texture_ptrs }
    }

    fn resolve<'a>(&'a self, id: std::num::NonZeroU64) -> Option<&'a wgpu::Texture> {
        let wgpu_id = unsafe { std::mem::transmute(id) };
        self.texture_ptrs.get(&wgpu_id).map(|&ptr| unsafe { &*ptr })
    }
}
