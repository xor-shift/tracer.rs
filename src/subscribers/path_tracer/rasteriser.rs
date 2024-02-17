use super::geometry::GeometryElement;
use super::texture_set::TextureSet;
use super::vertex::Triangle;
use super::vertex::Vertex;

use crate::subscriber::*;
use crate::Application;

use cgmath::ElementWise;
use cgmath::Matrix;
use wgpu::util::DeviceExt;

pub struct Rasteriser {
    pipeline: wgpu::RenderPipeline,

    vertex_buffer: wgpu::Buffer,

    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,

    pub texture_set: TextureSet,

    uniform_bind_group_layout: wgpu::BindGroupLayout,
    uniform_bind_group: wgpu::BindGroup,
}

const CBL: [f32; 3] = [-3.5, -3.5, -20.];
const CTR: [f32; 3] = [3.5, 2.5, 20.];

#[rustfmt::skip]
pub(crate) const TRIANGLES: [Triangle; 26] = [
    /*// light 1
    Triangle::new([[-3., 2.4, 15.], [-1., 2.4, 15.], [-1., 2.4, 11.25]], 6),
    Triangle::new([[-1., 2.4, 11.25], [-3., 2.4, 11.25], [-3., 2.4, 15.]], 6),

    // light 2
    Triangle::new([[1., 2.4, 15.], [3., 2.4, 15.], [3., 2.4, 11.25]], 7),
    Triangle::new([[3., 2.4, 11.25], [1., 2.4, 11.25], [1., 2.4, 15.]], 7),

    // light 3
    Triangle::new([[-1.25, 2.4, 12.], [1.25, 2.4, 12.], [1.25, 2.4, 8.25]], 8),
    Triangle::new([[1.25, 2.4, 8.25], [-1.25, 2.4, 8.25], [-1.25, 2.4, 12.]], 8),*/

    // light
    // triangle
    /*Vertex { position: [-1.25, 2.4, 15.], material: 0 },
    Vertex { position: [1.25, 2.4, 15.], material: 0 },
    Vertex { position: [1.25, 2.4, 11.25], material: 0 },
    // triangle
    Vertex { position: [1.25, 2.4, 11.25], material: 0 },
    Vertex { position: [-1.25, 2.4, 11.25], material: 0 },
    Vertex { position: [-1.25, 2.4, 15.], material: 0 },*/

    Triangle::new([[-1.25, 2.4, 15.], [1.25, 2.4, 15.], [1.25, 2.4, 11.25]], 0),
    Triangle::new([[1.25, 2.4, 11.25], [-1.25, 2.4, 11.25], [-1.25, 2.4, 15.]], 0),

    // mirror prism (bounding box: [-2.65, -2.5, 16.6], [-0.85, -0.7, 18.4])
    Triangle::new([[-2.65, -2.5, 16.6], [-2.65, -2.5, 18.4], [-0.85, -2.5, 18.4]], 1), // bottom 1
    Triangle::new([[-0.85, -2.5, 18.4], [-0.85, -2.5, 16.6], [-2.65, -2.5, 16.6]], 1), // bottom 2
    Triangle::new([[-2.65, -2.5, 18.4], [-2.65, -2.5, 16.6], [-1.75, -0.7, 17.5]], 1), // west
    Triangle::new([[-2.65, -2.5, 16.6], [-0.85, -2.5, 16.6], [-1.75, -0.7, 17.5]], 1), // south
    Triangle::new([[-0.85, -2.5, 16.6], [-0.85, -2.5, 18.4], [-1.75, -0.7, 17.5]], 1), // east
    Triangle::new([[-0.85, -2.5, 18.4], [-2.65, -2.5, 18.4], [-1.75, -0.7, 17.5]], 1), // north

    // glass prism (bounding box: [0.85, -2.3, 15.6], [2.65, -0.5, 17.4])
    Triangle::new([[-2.65 + 3.5, -2.5 + 0.2, 16.6 + -1.], [-2.65 + 3.5, -2.5 + 0.2, 18.4 + -1.], [-0.85 + 3.5, -2.5 + 0.2, 18.4 + -1.]], 2), // bottom 1
    Triangle::new([[-0.85 + 3.5, -2.5 + 0.2, 18.4 + -1.], [-0.85 + 3.5, -2.5 + 0.2, 16.6 + -1.], [-2.65 + 3.5, -2.5 + 0.2, 16.6 + -1.]], 2), // bottom 2
    Triangle::new([[-2.65 + 3.5, -2.5 + 0.2, 18.4 + -1.], [-2.65 + 3.5, -2.5 + 0.2, 16.6 + -1.], [-1.75 + 3.5 + 0., -0.7 + 0.2 + -0.3, 17.5 + -1. + 0.]], 2), // west
    Triangle::new([[-2.65 + 3.5, -2.5 + 0.2, 16.6 + -1.], [-0.85 + 3.5, -2.5 + 0.2, 16.6 + -1.], [-1.75 + 3.5 + 0., -0.7 + 0.2 + -0.3, 17.5 + -1. + 0.]], 2), // south
    Triangle::new([[-0.85 + 3.5, -2.5 + 0.2, 16.6 + -1.], [-0.85 + 3.5, -2.5 + 0.2, 18.4 + -1.], [-1.75 + 3.5 + 0., -0.7 + 0.2 + -0.3, 17.5 + -1. + 0.]], 2), // east
    Triangle::new([[-0.85 + 3.5, -2.5 + 0.2, 18.4 + -1.], [-2.65 + 3.5, -2.5 + 0.2, 18.4 + -1.], [-1.75 + 3.5 + 0., -0.7 + 0.2 + -0.3, 17.5 + -1. + 0.]], 2), // north

    // front wall
    Triangle::new([[CBL[0], CTR[1], CTR[2]], [CBL[0], CBL[1], CTR[2]], [CTR[0], CBL[1], CTR[2]]], 3),
    Triangle::new([[CTR[0], CBL[1], CTR[2]], [CTR[0], CTR[1], CTR[2]], [CBL[0], CTR[1], CTR[2]]], 3),

    // back wall
    Triangle::new([[CBL[0], CTR[1], CBL[2]], [CBL[0], CBL[1], CBL[2]], [CTR[0], CBL[1], CBL[2]]], 3),
    Triangle::new([[CTR[0], CBL[1], CBL[2]], [CTR[0], CTR[1], CBL[2]], [CBL[0], CTR[1], CBL[2]]], 3),

    // right wall
    Triangle::new([[CTR[0], CTR[1], CTR[2]], [CTR[0], CBL[1], CTR[2]], [CTR[0], CBL[1], CBL[2]]], 5),
    Triangle::new([[CTR[0], CBL[1], CBL[2]], [CTR[0], CTR[1], CBL[2]], [CTR[0], CTR[1], CTR[2]]], 5),

    // ceiling
    Triangle::new([[CBL[0], CTR[1], CTR[2]], [CTR[0], CTR[1], CTR[2]], [CTR[0], CTR[1], CBL[2]]], 3),
    Triangle::new([[CTR[0], CTR[1], CBL[2]], [CBL[0], CTR[1], CBL[2]], [CBL[0], CTR[1], CTR[2]]], 3),

    // left wall
    Triangle::new([[CBL[0], CTR[1], CTR[2]], [CBL[0], CBL[1], CTR[2]], [CBL[0], CBL[1], CBL[2]]], 4),
    Triangle::new([[CBL[0], CBL[1], CBL[2]], [CBL[0], CTR[1], CBL[2]], [CBL[0], CTR[1], CTR[2]]], 4),

    // floor
    Triangle::new([[CBL[0], CBL[1], CTR[2]], [CTR[0], CBL[1], CTR[2]], [CTR[0], CBL[1], CBL[2]]], 3),
    Triangle::new([[CTR[0], CBL[1], CBL[2]], [CBL[0], CBL[1], CBL[2]], [CBL[0], CBL[1], CTR[2]]], 3),
];

impl Rasteriser {
    fn depth_texture(app: &mut Application, extent: (u32, u32)) -> (wgpu::Texture, wgpu::TextureView) {
        let extent = wgpu::Extent3d {
            width: extent.0,
            height: extent.1,
            depth_or_array_layers: 1,
        };

        let texture = app.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tracer.rs rasteriser depth texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&std::default::Default::default());

        (texture, view)
    }

    pub fn new(app: &mut Application, state_buffer: &wgpu::Buffer) -> color_eyre::Result<Rasteriser> {
        let extent = app.window.inner_size().into();

        let (depth_texture, depth_texture_view) = Self::depth_texture(app, extent);

        let shader = app.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tracer.rs rasteriser shader"),
            source: wgpu::ShaderSource::Wgsl(std::fs::read_to_string("./shaders/out/rasteriser.wgsl").unwrap().as_str().into()),
        });

        let vertices = super::vertex::triangles_into_vertices(&TRIANGLES);

        let vertex_buffer = app.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let uniform_bind_group_layout = app.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tracer.rs rasteriser uniform bind group layout"),
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

        let uniform_bind_group = app.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tracer.rs rasteriser uniform bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: state_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = app.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tracer.rs main rasteriser pipeline layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = app.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tracer.rs rasteriser pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Uint,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: std::default::Default::default(),
                bias: std::default::Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let texture_set = TextureSet::new(extent, &app.device);

        return Ok(Self {
            pipeline,

            texture_set,
            depth_texture,
            depth_texture_view,

            vertex_buffer,

            uniform_bind_group_layout,
            uniform_bind_group,
        });
    }

    pub fn resize(&mut self, app: &mut Application, new_size: winit::dpi::PhysicalSize<u32>) {
        let (depth_texture, depth_texture_view) = Self::depth_texture(app, new_size.into());

        self.depth_texture = depth_texture;
        self.depth_texture_view = depth_texture_view;

        let texture_set = TextureSet::new(new_size.into(), &app.device);
        self.texture_set = texture_set;
    }

    pub fn render(&mut self, app: &mut Application, delta_time: std::time::Duration) {
        let mut encoder = app.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tracer.rs rasterisation encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tracer.rs rasteriser pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.texture_set.albedo.1,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0., g: 0., b: 0., a: 1. }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.texture_set.pack_normal_depth.1,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0., g: 0., b: 0., a: 1. }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.texture_set.pack_position_distance.1,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0., g: 0., b: 0., a: 1. }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.texture_set.object_indices.1,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0., g: 0., b: 0., a: 1. }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

            render_pass.draw(0..(TRIANGLES.len() * 3) as u32, 0..1);
        }

        let index = app.queue.submit(std::iter::once(encoder.finish()));
        app.device.poll(wgpu::Maintain::WaitForSubmissionIndex(index)).panic_on_timeout();
    }
}
