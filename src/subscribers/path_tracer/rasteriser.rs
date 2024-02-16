use super::geometry::GeometryElement;
use super::texture_set::TextureSet;
use super::vertex::Triangle;
use super::vertex::Vertex;

use crate::subscriber::*;
use crate::Application;

use cgmath::Matrix;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RawRasteriserUniform {
    camera: [[f32; 4]; 4],
    width: u32,
    height: u32,
    padding: [u32; 2],
}

pub struct RasteriserUniform {
    camera_position: cgmath::Point3<f32>,
    camera_rotation: cgmath::Vector3<f32>,
    dimensions: (u32, u32),
}

impl RasteriserUniform {
    pub fn new(dimensions: (u32, u32)) -> RasteriserUniform {
        Self {
            camera_position: cgmath::point3(0., 0., 0.),
            camera_rotation: cgmath::vec3(0., 0., 0.),
            dimensions,
        }
    }

    pub fn update(&mut self, app: &mut Application, delta_time: std::time::Duration) {
        let mut pending_movement = cgmath::vec3(0., 0., 0.);
        //let mut pending_rotation = cgmath::vec3(0., 0., 0.);

        if app.input_store.is_pressed(winit::keyboard::Key::Character("w".into())) {
            pending_movement[2] += 1.;
        }
        if app.input_store.is_pressed(winit::keyboard::Key::Character("a".into())) {
            pending_movement[0] += 1.; // TODO
        }
        if app.input_store.is_pressed(winit::keyboard::Key::Character("s".into())) {
            pending_movement[2] -= 1.;
        }
        if app.input_store.is_pressed(winit::keyboard::Key::Character("d".into())) {
            pending_movement[0] -= 1.; // TODO
        }
        if app.input_store.is_pressed(winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space)) {
            pending_movement[1] += 1.;
        }
        if app.input_store.is_pressed(winit::keyboard::Key::Named(winit::keyboard::NamedKey::Shift)) {
            pending_movement[1] -= 1.;
        }

        self.camera_position += pending_movement * delta_time.as_secs_f32() * 10.;
        self.dimensions = app.window.inner_size().into();
    }

    pub fn generate(&self) -> RawRasteriserUniform {
        /*#[rustfmt::skip]
        let matrix = cgmath::Matrix4::<f32>::new(
            1., 0., 0., self.camera_position.x,
            0., 1., 0., self.camera_position.y,
            0., 0., 1., self.camera_position.z,
            0., 0., 0., 1.,
        ).transpose();*/

        let rotation = // a
            cgmath::Matrix3::from_angle_x(cgmath::Deg(self.camera_rotation[1])) *
            cgmath::Matrix3::from_angle_y(cgmath::Deg(self.camera_rotation[0])) *
            cgmath::Matrix3::from_angle_z(cgmath::Deg(self.camera_rotation[2]));

        let look_at = rotation * cgmath::vec3(0., 0., 1.);
        let look_at = self.camera_position + look_at;

        let view = cgmath::Matrix4::look_at_rh(self.camera_position, look_at, cgmath::vec3(0., 1., 0.));
        let proj = cgmath::perspective(cgmath::Deg(30.), self.dimensions.0 as f32 / self.dimensions.1 as f32, 0.1, 1000.);

        let matrix = proj * view;

        RawRasteriserUniform {
            camera: matrix.into(),
            width: self.dimensions.0,
            height: self.dimensions.1,
            padding: [0; 2],
        }
    }
}

pub struct Rasteriser {
    pipeline: wgpu::RenderPipeline,

    vertex_buffer: wgpu::Buffer,

    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,

    pub texture_set: TextureSet,
    pub geometry_buffer: wgpu::Buffer,
    uniform: RasteriserUniform,
    uniform_buffer: wgpu::Buffer,

    uniform_bind_group_layout: wgpu::BindGroupLayout,
    uniform_bind_group: wgpu::BindGroup,

    geometry_bind_group_layout: wgpu::BindGroupLayout,
    geometry_bind_group: wgpu::BindGroup,
}

const CBL: [f32; 3] = [-3.5, -3.5, -20.];
const CTR: [f32; 3] = [3.5, 2.5, 20.];

#[rustfmt::skip]
const TRIANGLES: [Triangle; 30] = [
    // light 1
    Triangle { vertices: [[-3., 2.4, 15.], [-1., 2.4, 15.], [-1., 2.4, 11.25]], material: 6 },
    Triangle { vertices: [[-1., 2.4, 11.25], [-3., 2.4, 11.25], [-3., 2.4, 15.]], material: 6 },

    // light 2
    Triangle { vertices: [[1., 2.4, 15.], [3., 2.4, 15.], [3., 2.4, 11.25]], material: 7 },
    Triangle { vertices: [[3., 2.4, 11.25], [1., 2.4, 11.25], [1., 2.4, 15.]], material: 7 },

    // light 3
    Triangle { vertices: [[-1.25, 2.4, 12.], [1.25, 2.4, 12.], [1.25, 2.4, 8.25]], material: 8 },
    Triangle { vertices: [[1.25, 2.4, 8.25], [-1.25, 2.4, 8.25], [-1.25, 2.4, 12.]], material: 8 },

    // light
    // triangle
    /*Vertex { position: [-1.25, 2.4, 15.], material: 0 },
    Vertex { position: [1.25, 2.4, 15.], material: 0 },
    Vertex { position: [1.25, 2.4, 11.25], material: 0 },
    // triangle
    Vertex { position: [1.25, 2.4, 11.25], material: 0 },
    Vertex { position: [-1.25, 2.4, 11.25], material: 0 },
    Vertex { position: [-1.25, 2.4, 15.], material: 0 },*/

    // mirror prism (bounding box: [-2.65, -2.5, 16.6], [-0.85, -0.7, 18.4])
    // bottom 1
    Triangle { vertices: [[-2.65, -2.5, 16.6], [-2.65, -2.5, 18.4], [-0.85, -2.5, 18.4]], material: 1 },
    // bottom 2
    Triangle { vertices: [[-0.85, -2.5, 18.4], [-0.85, -2.5, 16.6], [-2.65, -2.5, 16.6]], material: 1 },
    // west
    Triangle { vertices: [[-2.65, -2.5, 18.4], [-2.65, -2.5, 16.6], [-1.75, -0.7, 17.5]], material: 1 },
    // south
    Triangle { vertices: [[-2.65, -2.5, 16.6], [-0.85, -2.5, 16.6], [-1.75, -0.7, 17.5]], material: 1 },
    // east
    Triangle { vertices: [[-0.85, -2.5, 16.6], [-0.85, -2.5, 18.4], [-1.75, -0.7, 17.5]], material: 1 },
    // north
    Triangle { vertices: [[-0.85, -2.5, 18.4], [-2.65, -2.5, 18.4], [-1.75, -0.7, 17.5]], material: 1 },

    // glass prism (bounding box: [0.85, -2.3, 15.6], [2.65, -0.5, 17.4])
    // bottom 1
    Triangle { vertices: [[-2.65 + 3.5, -2.5 + 0.2, 16.6 + -1.], [-2.65 + 3.5, -2.5 + 0.2, 18.4 + -1.], [-0.85 + 3.5, -2.5 + 0.2, 18.4 + -1.]], material: 2 },
    // bottom 2
    Triangle { vertices: [[-0.85 + 3.5, -2.5 + 0.2, 18.4 + -1.], [-0.85 + 3.5, -2.5 + 0.2, 16.6 + -1.], [-2.65 + 3.5, -2.5 + 0.2, 16.6 + -1.]], material: 2 },
    // west
    Triangle { vertices: [[-2.65 + 3.5, -2.5 + 0.2, 18.4 + -1.], [-2.65 + 3.5, -2.5 + 0.2, 16.6 + -1.], [-1.75 + 3.5 + 0., -0.7 + 0.2 + -0.3, 17.5 + -1. + 0.]], material: 2 },
    // south
    Triangle { vertices: [[-2.65 + 3.5, -2.5 + 0.2, 16.6 + -1.], [-0.85 + 3.5, -2.5 + 0.2, 16.6 + -1.], [-1.75 + 3.5 + 0., -0.7 + 0.2 + -0.3, 17.5 + -1. + 0.]], material: 2 },
    // east
    Triangle { vertices: [[-0.85 + 3.5, -2.5 + 0.2, 16.6 + -1.], [-0.85 + 3.5, -2.5 + 0.2, 18.4 + -1.], [-1.75 + 3.5 + 0., -0.7 + 0.2 + -0.3, 17.5 + -1. + 0.]], material: 2 },
    // north
    Triangle { vertices: [[-0.85 + 3.5, -2.5 + 0.2, 18.4 + -1.], [-2.65 + 3.5, -2.5 + 0.2, 18.4 + -1.], [-1.75 + 3.5 + 0., -0.7 + 0.2 + -0.3, 17.5 + -1. + 0.]], material: 2 },

    // front wall
    Triangle { vertices: [[CBL[0], CTR[1], CTR[2]], [CBL[0], CBL[1], CTR[2]], [CTR[0], CBL[1], CTR[2]]], material: 3 },
    Triangle { vertices: [[CTR[0], CBL[1], CTR[2]], [CTR[0], CTR[1], CTR[2]], [CBL[0], CTR[1], CTR[2]]], material: 3 },

    // back wall
    Triangle { vertices: [[CBL[0], CTR[1], CBL[2]], [CBL[0], CBL[1], CBL[2]], [CTR[0], CBL[1], CBL[2]]], material: 3 },
    Triangle { vertices: [[CTR[0], CBL[1], CBL[2]], [CTR[0], CTR[1], CBL[2]], [CBL[0], CTR[1], CBL[2]]], material: 3 },

    // right wall
    Triangle { vertices: [[CTR[0], CTR[1], CTR[2]], [CTR[0], CBL[1], CTR[2]], [CTR[0], CBL[1], CBL[2]]], material: 5 },
    Triangle { vertices: [[CTR[0], CBL[1], CBL[2]], [CTR[0], CTR[1], CBL[2]], [CTR[0], CTR[1], CTR[2]]], material: 5 },

    // ceiling
    Triangle { vertices: [[CBL[0], CTR[1], CTR[2]], [CTR[0], CTR[1], CTR[2]], [CTR[0], CTR[1], CBL[2]]], material: 3 },
    Triangle { vertices: [[CTR[0], CTR[1], CBL[2]], [CBL[0], CTR[1], CBL[2]], [CBL[0], CTR[1], CTR[2]]], material: 3 },

    // left wall
    Triangle { vertices: [[CBL[0], CTR[1], CTR[2]], [CBL[0], CBL[1], CTR[2]], [CBL[0], CBL[1], CBL[2]]], material: 4 },
    Triangle { vertices: [[CBL[0], CBL[1], CBL[2]], [CBL[0], CTR[1], CBL[2]], [CBL[0], CTR[1], CTR[2]]], material: 4 },

    // floor
    Triangle { vertices: [[CBL[0], CBL[1], CTR[2]], [CTR[0], CBL[1], CTR[2]], [CTR[0], CBL[1], CBL[2]]], material: 3 },
    Triangle { vertices: [[CTR[0], CBL[1], CBL[2]], [CBL[0], CBL[1], CBL[2]], [CBL[0], CBL[1], CTR[2]]], material: 3 },
];

const VERTICES: [Vertex; 90] = super::vertex::triangles_into_vertices(&TRIANGLES);

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

        /*let sampler = app.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });*/

        (texture, view)
    }

    pub fn new(app: &mut Application) -> color_eyre::Result<Rasteriser> {
        let extent = app.window.inner_size().into();

        let (depth_texture, depth_texture_view) = Self::depth_texture(app, extent);

        let shader = app.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tracer.rs rasteriser shader"),
            source: wgpu::ShaderSource::Wgsl(std::fs::read_to_string("./shaders/out/rasteriser.wgsl").unwrap().as_str().into()),
        });

        let vertex_buffer = app.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let uniform = RasteriserUniform::new(extent);
        let uniform_buffer = app.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tracer.rs rasteriser camera buffer"),
            contents: bytemuck::cast_slice(&[uniform.generate()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let geometry_buffer = app.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tracer.rs rasteriser geometry buffer"),
            size: extent.0 as u64 * extent.1 as u64 * std::mem::size_of::<GeometryElement>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let geometry_bind_group_layout = app.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tracer.rs rasteriser geometry buffer bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let geometry_bind_group = app.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tracer.rs rasteriser geometry buffer bind group"),
            layout: &geometry_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: geometry_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = app.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tracer.rs main rasteriser pipeline layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &geometry_bind_group_layout],
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
            geometry_buffer,
            depth_texture,
            depth_texture_view,

            vertex_buffer,

            uniform,
            uniform_buffer,

            uniform_bind_group_layout,
            uniform_bind_group,

            geometry_bind_group_layout,
            geometry_bind_group,
        });
    }

    pub fn resize(&mut self, app: &mut Application, new_size: winit::dpi::PhysicalSize<u32>) {
        let (depth_texture, depth_texture_view) = Self::depth_texture(app, new_size.into());

        let geometry_buffer = app.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tracer.rs rasteriser geometry buffer"),
            size: new_size.width as u64 * new_size.height as u64 * std::mem::size_of::<GeometryElement>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        self.depth_texture = depth_texture;
        self.geometry_buffer = geometry_buffer;
        self.depth_texture_view = depth_texture_view;
        self.uniform.dimensions = new_size.into();

        let geometry_bind_group = app.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tracer.rs rasteriser geometry buffer bind group"),
            layout: &self.geometry_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.geometry_buffer.as_entire_binding(),
            }],
        });

        self.geometry_bind_group = geometry_bind_group;

        let texture_set = TextureSet::new(new_size.into(), &app.device);
        self.texture_set = texture_set;
    }

    pub fn render(&mut self, app: &mut Application, view: &wgpu::TextureView, _encoder: &mut wgpu::CommandEncoder, delta_time: std::time::Duration) {
        let mut encoder = app.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tracer.rs rasterisation encoder"),
        });

        {
            self.uniform.update(app, delta_time);
            let raw_uniform = self.uniform.generate();
            app.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[raw_uniform]));

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
            render_pass.set_bind_group(1, &self.geometry_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

            render_pass.draw(0..VERTICES.len() as u32, 0..1);
        }

        let index = app.queue.submit(std::iter::once(encoder.finish()));
        app.device.poll(wgpu::Maintain::WaitForSubmissionIndex(index)).panic_on_timeout();
    }
}
