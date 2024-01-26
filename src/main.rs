// #![feature(adt_const_params)]
#![feature(auto_traits)]
#![feature(const_trait_impl)]
#![feature(generic_const_exprs)]
#![feature(let_chains)]
#![feature(macro_metavar_expr)]
#![feature(negative_impls)]
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

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct RawMainUniform {
    pub frame_no: u32,
    pub current_instant: f32,
}

impl From<MainUniform> for RawMainUniform {
    fn from(value: MainUniform) -> Self {
        Self {
            frame_no: value.frame_no,
            current_instant: (std::time::Instant::now() - value.started_at).as_secs_f32(),
        }
    }
}

#[derive(Copy, Clone)]
struct MainUniform {
    pub frame_no: u32,
    pub started_at: std::time::Instant,
}

impl MainUniform {
    pub fn new() -> Self {
        MainUniform {
            frame_no: 0,
            started_at: std::time::Instant::now(),
        }
    }
}

struct ComputeTest {
    render_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,

    bind_group_layout_main: wgpu::BindGroupLayout,
    bind_group_main: wgpu::BindGroup,

    bind_group_layout_compute_textures: wgpu::BindGroupLayout,
    bind_group_compute_textures: wgpu::BindGroup,

    tex_dims: (u32, u32),
    texture_0: wgpu::Texture,
    texture_0_view: wgpu::TextureView,
    texture_1: wgpu::Texture,
    texture_1_view: wgpu::TextureView,

    num_vertices: u32,
    vertex_buffer: wgpu::Buffer,
    num_indices: u32,
    index_buffer: wgpu::Buffer,

    uniform_main: MainUniform,
    uniform_buffer_main: wgpu::Buffer,
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

    fn generate_textures(app: &mut gfx::Application, extent: (u32, u32)) -> [(wgpu::Texture, wgpu::TextureView); 2] {
        let texture_desc = wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: extent.0,
                height: extent.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };

        let texture_0 = app.device.create_texture(&texture_desc);
        let texture_0_view = texture_0.create_view(&std::default::Default::default());
        let texture_1 = app.device.create_texture(&texture_desc);
        let texture_1_view = texture_1.create_view(&std::default::Default::default());

        [(texture_0, texture_0_view), (texture_1, texture_1_view)]
    }

    fn new(app: &mut gfx::Application) -> Box<dyn gfx::Subscriber> {
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
            source: wgpu::ShaderSource::Wgsl(std::fs::read_to_string("./shaders/main.wgsl").unwrap().as_str().into()),
        });

        let shader_compute = app.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tracer.rs rendering shader"),
            source: wgpu::ShaderSource::Wgsl(std::fs::read_to_string("./shaders/compute.wgsl").unwrap().as_str().into()),
        });

        let [(texture_0, texture_0_view), (texture_1, texture_1_view)] = Self::generate_textures(app, (512, 512));

        let bind_group_layout_main = app.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group_layout_compute_textures = app.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, ..default_tex_bind }, //
                wgpu::BindGroupLayoutEntry { binding: 1, ..default_tex_bind },
            ],
        });

        let render_pipeline_layout = app.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tracer.rs main pipeline layout"),
            bind_group_layouts: &[&bind_group_layout_main, &bind_group_layout_compute_textures],
            push_constant_ranges: &[],
        });

        let compute_pipeline_layout = app.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tracer.rs main pipeline layout"),
            bind_group_layouts: &[&bind_group_layout_main, &bind_group_layout_compute_textures],
            push_constant_ranges: &[],
        });

        let uniform_main = MainUniform::new();
        let uniform_buffer_main = app.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("main compute uniform"),
            contents: bytemuck::cast_slice(&[std::convert::Into::<RawMainUniform>::into(uniform_main)]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let vertex_buffer = app.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&Self::VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = app.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&Self::INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let bind_group_main = app.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout_main,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer_main.as_entire_binding(),
            }],
        });

        let bind_group_compute_textures = app.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout_compute_textures,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_0_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_1_view),
                },
            ],
        });

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

            bind_group_layout_compute_textures,
            bind_group_compute_textures,

            tex_dims: (512, 512),
            texture_0,
            texture_0_view,
            texture_1,
            texture_1_view,

            num_vertices: Self::VERTICES.len() as u32,
            vertex_buffer,
            num_indices: Self::INDICES.len() as u32,
            index_buffer,

            uniform_main,
            uniform_buffer_main,
        };

        Box::new(this)
    }
}

impl gfx::Subscriber for ComputeTest {
    fn resize(&mut self, app: &mut gfx::Application, new_size: winit::dpi::PhysicalSize<u32>) {
        let [(texture_0, texture_0_view), (texture_1, texture_1_view)] = Self::generate_textures(app, (new_size.width, new_size.height));

        let new_bind_group_compute_textures = app.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout_compute_textures,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_0_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_1_view),
                },
            ],
        });

        self.tex_dims = (new_size.width, new_size.height);

        self.texture_0 = texture_0;
        self.texture_0_view = texture_0_view;

        self.texture_1 = texture_1;
        self.texture_1_view = texture_1_view;

        self.bind_group_compute_textures = new_bind_group_compute_textures;
    }

    fn render(&mut self, app: &mut gfx::Application, view: &wgpu::TextureView, encoder: &mut wgpu::CommandEncoder, delta_time: std::time::Duration) {
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
            render_pass.set_bind_group(1, &self.bind_group_compute_textures, &[]);
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tracer.rs compute pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group_main, &[]);
            compute_pass.set_bind_group(1, &self.bind_group_compute_textures, &[]);
            compute_pass.dispatch_workgroups(self.tex_dims.0, self.tex_dims.1, 1);
        }

        self.uniform_main.frame_no += 1;
        app.queue.write_buffer(&self.uniform_buffer_main, 0, bytemuck::cast_slice(&[std::convert::Into::<RawMainUniform>::into(self.uniform_main)]));
    }
}

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

    let subscriber = ComputeTest::new(&mut app);
    app.add_subscriber(subscriber);

    app.run().await.expect("app exited with an error");
}
