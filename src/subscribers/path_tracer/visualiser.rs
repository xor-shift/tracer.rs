use super::state::State;
use super::texture_set::TextureSet;

use crate::subscriber::*;
use crate::Application;

use wgpu::util::DeviceExt;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct VisualiserUniform {
    pub width: u32,
    pub height: u32,
    pub mode: i32,
}

pub struct Visualiser {
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    texture_set_bgl: wgpu::BindGroupLayout,
    texture_set_bg: wgpu::BindGroup,
    texture_set_bg_swapped: wgpu::BindGroup,

    pipeline: wgpu::RenderPipeline,
}

impl Visualiser {
    fn make_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let tex_bind = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        let tex_bind_uint = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        let desc = wgpu::BindGroupLayoutDescriptor {
            label: Some("tracer.rs visualiser texture set bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, ..tex_bind },      // path trace result
                wgpu::BindGroupLayoutEntry { binding: 1, ..tex_bind_uint }, // geometry pack 0
                wgpu::BindGroupLayoutEntry { binding: 2, ..tex_bind_uint }, // geometry pack 1
                wgpu::BindGroupLayoutEntry { binding: 3, ..tex_bind },      // denoise_0
                wgpu::BindGroupLayoutEntry { binding: 4, ..tex_bind },      // denoise_1
            ],
        };

        device.create_bind_group_layout(&desc)
    }

    fn make_bg(texture_set: &TextureSet, device: &wgpu::Device, layout: &wgpu::BindGroupLayout, swapped: bool) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tracer.rs visualiser texture set bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(if swapped { &texture_set.ray_trace_1.1 } else { &texture_set.ray_trace_0.1 }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_set.geometry_pack_0.1),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&texture_set.geometry_pack_1.1),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&texture_set.denoise_0.1),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&texture_set.denoise_1.1),
                },
            ],
        })
    }

    pub fn new(app: &mut Application, texture_set: &TextureSet) -> color_eyre::Result<Visualiser> {
        let shader = app.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tracer.rs rendering shader"),
            source: wgpu::ShaderSource::Wgsl(std::fs::read_to_string("./shaders/out/main.wgsl").unwrap().as_str().into()),
        });

        let uniform_buffer: wgpu::Buffer = app.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("main uniform"),
            contents: bytemuck::cast_slice(&[VisualiserUniform {
                width: app.window.inner_size().width,
                height: app.window.inner_size().height,
                mode: 0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout = app.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let uniform_bind_group = app.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let texture_set_bgl = Self::make_bgl(&app.device);
        let texture_set_bg = Self::make_bg(texture_set, &app.device, &texture_set_bgl, false);
        let texture_set_bg_swapped = Self::make_bg(texture_set, &app.device, &texture_set_bgl, true);

        let render_pipeline_layout = app.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tracer.rs main pipeline layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &texture_set_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = app.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tracer.rs main render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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

        Ok(Self {
            uniform_buffer,
            uniform_bind_group,

            texture_set_bgl,
            texture_set_bg,
            texture_set_bg_swapped,

            pipeline,
        })
    }

    pub fn resize(&mut self, app: &mut Application, new_size: winit::dpi::PhysicalSize<u32>, texture_set: &TextureSet) {
        let texture_set_bg = Self::make_bg(texture_set, &app.device, &self.texture_set_bgl, false);
        let texture_set_bg_swapped = Self::make_bg(texture_set, &app.device, &self.texture_set_bgl, true);
        self.texture_set_bg = texture_set_bg;
        self.texture_set_bg_swapped = texture_set_bg_swapped;
    }

    pub fn render(&mut self, app: &mut Application, view: &wgpu::TextureView, encoder: &mut wgpu::CommandEncoder, vis_mode: i32, state: &State) {
        let new_uniform = VisualiserUniform {
            width: app.window.inner_size().width,
            height: app.window.inner_size().height,
            mode: vis_mode,
        };
        app.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[new_uniform]));

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

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_bind_group(1, if state.should_swap_buffers() { &self.texture_set_bg_swapped } else { &self.texture_set_bg }, &[]);
        //render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        //render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        render_pass.draw(0..6, 0..1);
    }
}
