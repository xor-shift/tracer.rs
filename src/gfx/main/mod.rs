mod geometry;
mod noise_texture;
mod texture_set;
mod uniform;

use super::vertex::Vertex;
use wgpu::util::DeviceExt;

use geometry::GeometryElement;
use noise_texture::NoiseTexture;
use texture_set::TextureSet;
use uniform::RawMainUniform;
use uniform::UniformGenerator;

pub struct ComputeTest {
    render_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,

    bind_group_layout_main: wgpu::BindGroupLayout,
    bind_group_main: wgpu::BindGroup,

    texture_set_layout_cs: wgpu::BindGroupLayout,
    texture_set_layout_fs: wgpu::BindGroupLayout,
    tex_dims: (u32, u32),
    texture_set_bind_groups_cs: [wgpu::BindGroup; 2],
    texture_set_bind_groups_fs: [wgpu::BindGroup; 2],
    texture_sets: [TextureSet; 2],

    num_vertices: u32,
    vertex_buffer: wgpu::Buffer,
    num_indices: u32,
    index_buffer: wgpu::Buffer,

    uniform_generator: UniformGenerator,
    uniform_buffer_main: wgpu::Buffer,

    noise: NoiseTexture,
}

impl ComputeTest {
    #[rustfmt::skip]
    const VERTICES: [Vertex; 4] = [
        Vertex { position: [-1., 1., 0.], tex_coords: [0., 0.] },
        Vertex { position: [-1., -1., 0.], tex_coords: [0., 1.] },
        Vertex { position: [1., -1., 0.], tex_coords: [1., 1.] },
        Vertex { position: [1., 1., 0.], tex_coords: [1., 0.] },
    ];

    #[rustfmt::skip]
    const INDICES: [u16; 6] = [
        0, 1, 2,
        2, 3, 0,
    ];

    fn generate_textures(app: &mut super::Application, extent: (u32, u32)) -> [TextureSet; 2] { [TextureSet::new(extent, &app.device), TextureSet::new(extent, &app.device)] }

    pub fn new(app: &mut super::Application) -> Box<dyn super::Subscriber> {
        let default_tex_bind = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::ReadWrite,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };

        let shader_render = app.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tracer.rs rendering shader"),
            source: wgpu::ShaderSource::Wgsl(std::fs::read_to_string("./shaders/out/main.wgsl").unwrap().as_str().into()),
        });

        let shader_compute = app.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tracer.rs rendering shader"),
            source: wgpu::ShaderSource::Wgsl(std::fs::read_to_string("./shaders/out/compute.wgsl").unwrap().as_str().into()),
        });

        let bind_group_layout_main = app.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: wgpu::TextureFormat::Rgba32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let texture_set_layout_fs = TextureSet::bind_group_layout_fs(&app.device);
        let render_pipeline_layout = app.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tracer.rs main pipeline layout"),
            bind_group_layouts: &[&bind_group_layout_main, &texture_set_layout_fs],
            push_constant_ranges: &[],
        });

        let texture_set_layout_cs = TextureSet::bind_group_layout_cs(&app.device);
        let compute_pipeline_layout = app.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tracer.rs main pipeline layout"),
            bind_group_layouts: &[&bind_group_layout_main, &texture_set_layout_cs],
            push_constant_ranges: &[],
        });

        let uniform_generator = UniformGenerator::new();
        let uniform_buffer_main: wgpu::Buffer = app.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("main uniform"),
            contents: bytemuck::cast_slice(&[std::convert::Into::<RawMainUniform>::into(uniform_generator.generate())]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let vertex_buffer = app.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex buffer"),
            contents: bytemuck::cast_slice(&Self::VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = app.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index buffer"),
            contents: bytemuck::cast_slice(&Self::INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let noise = NoiseTexture::new((128, 128), &app.device, &app.queue, Some("tracer.rs noise texture"));

        let bind_group_main = app.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout_main,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer_main.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&noise.view),
                },
            ],
        });

        let texture_sets = Self::generate_textures(app, (512, 512));

        let texture_set_bind_groups_cs = [
            texture_sets[0].bind_group_cs(&app.device, &texture_set_layout_cs), //
            texture_sets[1].bind_group_cs(&app.device, &texture_set_layout_cs),
        ];

        let texture_set_bind_groups_fs = [
            texture_sets[0].bind_group_fs(&app.device, &texture_set_layout_fs), //
            texture_sets[1].bind_group_fs(&app.device, &texture_set_layout_fs),
        ];

        let render_pipeline = app.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tracer.rs main render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_render,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_render,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: app.config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
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

        let compute_pipeline = app.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tracer.rs main compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_compute,
            entry_point: "cs_main",
        });

        let this = Self {
            render_pipeline,
            compute_pipeline,

            bind_group_layout_main,
            bind_group_main,

            texture_set_layout_cs,
            texture_set_layout_fs,
            tex_dims: (512, 512),
            texture_set_bind_groups_cs,
            texture_set_bind_groups_fs,
            texture_sets,

            num_vertices: Self::VERTICES.len() as u32,
            vertex_buffer,
            num_indices: Self::INDICES.len() as u32,
            index_buffer,

            uniform_generator,
            uniform_buffer_main,

            noise,
        };

        Box::new(this)
    }
}

impl super::Subscriber for ComputeTest {
    fn handle_event<'a>(&mut self, app: &'a mut super::Application, event: &winit::event::Event<'a, ()>) -> super::EventHandlingResult {
        match event {
            winit::event::Event::WindowEvent { window_id, event } if *window_id == app.window.id() => match event {
                winit::event::WindowEvent::KeyboardInput { device_id, input, is_synthetic } if input.state == winit::event::ElementState::Pressed => {
                    let action_opt = input.virtual_keycode.and_then(|code| match code {
                        winit::event::VirtualKeyCode::Q => Some(-1),
                        winit::event::VirtualKeyCode::E => Some(1),
                        _ => None,
                    });

                    if let Some(action) = action_opt {
                        self.uniform_generator.visualisation_mode += action;
                        log::debug!("vis mode: {}", self.uniform_generator.visualisation_mode);
                        super::EventHandlingResult::Handled
                    } else {
                        super::EventHandlingResult::NotHandled
                    }
                }
                _ => super::EventHandlingResult::NotHandled,
            },
            _ => super::EventHandlingResult::NotHandled,
        }
    }

    fn resize(&mut self, app: &mut super::Application, new_size: winit::dpi::PhysicalSize<u32>) {
        let texture_sets = Self::generate_textures(app, (new_size.width, new_size.height));

        let texture_set_bind_groups_cs = [
            texture_sets[0].bind_group_cs(&app.device, &self.texture_set_layout_cs), //
            texture_sets[1].bind_group_cs(&app.device, &self.texture_set_layout_cs),
        ];

        let texture_set_bind_groups_fs = [
            texture_sets[0].bind_group_fs(&app.device, &self.texture_set_layout_fs), //
            texture_sets[1].bind_group_fs(&app.device, &self.texture_set_layout_fs),
        ];

        self.tex_dims = (new_size.width, new_size.height);
        self.texture_sets = texture_sets;

        self.texture_set_bind_groups_cs = texture_set_bind_groups_cs;
        self.texture_set_bind_groups_fs = texture_set_bind_groups_fs;

        self.uniform_generator.reset();
    }

    fn render(&mut self, app: &mut super::Application, view: &wgpu::TextureView, encoder: &mut wgpu::CommandEncoder, delta_time: std::time::Duration) {
        let raw_uniform = self.uniform_generator.frame_start();
        app.queue.write_buffer(&self.uniform_buffer_main, 0, bytemuck::cast_slice(&[raw_uniform]));

        let (back_buffer_fs, front_buffer_fs) = if self.uniform_generator.frame_no % 2 == 0 {
            (&self.texture_set_bind_groups_fs[0], &self.texture_set_bind_groups_fs[1])
        } else {
            (&self.texture_set_bind_groups_fs[1], &self.texture_set_bind_groups_fs[0])
        };

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tracer.rs render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group_main, &[]);
            render_pass.set_bind_group(1, front_buffer_fs, &[]);
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        let (back_buffer_cs, front_buffer_cs) = if self.uniform_generator.frame_no % 2 == 0 {
            (&self.texture_set_bind_groups_cs[0], &self.texture_set_bind_groups_cs[1])
        } else {
            (&self.texture_set_bind_groups_cs[1], &self.texture_set_bind_groups_cs[0])
        };

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tracer.rs compute pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group_main, &[]);
            compute_pass.set_bind_group(1, back_buffer_cs, &[]);

            let wg_dims = (8u32, 8u32);
            compute_pass.dispatch_workgroups(
                (self.tex_dims.0 + wg_dims.0 - 1) / wg_dims.0, //
                (self.tex_dims.1 + wg_dims.1 - 1) / wg_dims.1,
                1,
            );
        }

        self.uniform_generator.frame_end();
    }
}
