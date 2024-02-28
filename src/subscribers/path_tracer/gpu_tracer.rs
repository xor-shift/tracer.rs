use super::noise_texture::NoiseTexture;
use super::state::State;
use super::texture_set::TextureSet;

use crate::subscriber::*;
use crate::Application;

use wgpu::util::DeviceExt;
use wgpu::BindGroupDescriptor;

pub struct GPUTracer {
    bind_group_main: wgpu::BindGroup,

    texture_set_bgl: wgpu::BindGroupLayout,
    texture_set_bg: wgpu::BindGroup,
    texture_set_bg_swapped: wgpu::BindGroup,

    triangles_buffer: wgpu::Buffer,
    triangles_bgl: wgpu::BindGroupLayout,
    triangles_bg: wgpu::BindGroup,

    pipeline: wgpu::ComputePipeline,
}

impl GPUTracer {
    fn make_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let desc = wgpu::BindGroupLayoutDescriptor {
            label: Some("tracer.rs path tracer texture set bind group layout"),
            entries: &[
                TextureSet::make_bgle_storage_cs(0, true, wgpu::TextureFormat::Rgba8Unorm), // pt results
                TextureSet::make_bgle_regular_cs(1, true),                                  // previous frame pt results
                TextureSet::make_bgle_storage_cs(2, true, wgpu::TextureFormat::Rgba32Uint), // geometry pack 0
                TextureSet::make_bgle_storage_cs(3, true, wgpu::TextureFormat::Rgba32Uint), // geometry pack 1
                TextureSet::make_bgle_regular_cs(4, false),                                 // previous frame geometry pack 0
                TextureSet::make_bgle_regular_cs(5, false),                                 // previous frame geometry pack 1
            ],
        };

        device.create_bind_group_layout(&desc)
    }

    fn make_bg(texture_set: &TextureSet, device: &wgpu::Device, layout: &wgpu::BindGroupLayout, swap: bool) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tracer.rs path tracer texture set bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(if swap { &texture_set.ray_trace_1.1 } else { &texture_set.ray_trace_0.1 }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(if swap { &texture_set.ray_trace_0.1 } else { &texture_set.ray_trace_1.1 }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(if swap { &texture_set.geometry_pack_0_swap.1 } else { &texture_set.geometry_pack_0.1 }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(if swap { &texture_set.geometry_pack_1_swap.1 } else { &texture_set.geometry_pack_1.1 }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(if !swap { &texture_set.geometry_pack_0_swap.1 } else { &texture_set.geometry_pack_0.1 }),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(if !swap { &texture_set.geometry_pack_1_swap.1 } else { &texture_set.geometry_pack_1.1 }),
                },
            ],
        })
    }

    pub fn new(app: &mut Application, texture_set: &TextureSet, state_buffer: &wgpu::Buffer, old_state_buffer: &wgpu::Buffer) -> color_eyre::Result<GPUTracer> {
        let extent: (u32, u32) = app.window.inner_size().into();

        let shader = app.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tracer.rs rendering shader"),
            source: wgpu::ShaderSource::Wgsl(std::fs::read_to_string("./shaders/out/compute.wgsl")?.as_str().into()),
        });

        let noise: NoiseTexture = NoiseTexture::new((128, 128), &app.device, &app.queue, Some("tracer.rs noise texture"));

        let bind_group_layout_main = app.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        let bind_group_main = app.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout_main,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: old_state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&noise.view),
                },
            ],
        });

        let texture_set_bgl = Self::make_bgl(&app.device);
        let texture_set_bg = Self::make_bg(texture_set, &app.device, &texture_set_bgl, false);
        let texture_set_bg_swapped = Self::make_bg(texture_set, &app.device, &texture_set_bgl, true);

        let triangles_buffer: wgpu::Buffer = app.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tracer.rs pt triangle buffer"),
            contents: bytemuck::cast_slice(&[super::rasteriser::TRIANGLES]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let triangles_bgl = app.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let triangles_bg = app.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &triangles_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: triangles_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = app.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tracer.rs path tracer compute pipeline layout"),
            bind_group_layouts: &[&bind_group_layout_main, &texture_set_bgl, &triangles_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = app.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tracer.rs path tracer compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
        });

        Ok(Self {
            bind_group_main,

            texture_set_bgl,
            texture_set_bg,
            texture_set_bg_swapped,

            triangles_buffer,
            triangles_bgl,
            triangles_bg,

            pipeline,
        })
    }

    pub fn resize(&mut self, app: &mut Application, new_size: winit::dpi::PhysicalSize<u32>, texture_set: &TextureSet) {
        let texture_set_bg = Self::make_bg(texture_set, &app.device, &self.texture_set_bgl, false);
        let texture_set_bg_swapped = Self::make_bg(texture_set, &app.device, &self.texture_set_bgl, true);
        self.texture_set_bg = texture_set_bg;
        self.texture_set_bg_swapped = texture_set_bg_swapped;
    }

    pub fn render(&mut self, app: &mut Application, state: &State) {
        let mut encoder = app.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tracer.rs path tracer encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tracer.rs path tracer compute pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group_main, &[]);
            compute_pass.set_bind_group(1, if state.should_swap_buffers() { &self.texture_set_bg_swapped } else { &self.texture_set_bg }, &[]);
            compute_pass.set_bind_group(2, &self.triangles_bg, &[]);

            let wg_dims = (8u32, 8u32);
            compute_pass.dispatch_workgroups(
                (app.window.inner_size().width + wg_dims.0 - 1) / wg_dims.0, //
                (app.window.inner_size().height + wg_dims.1 - 1) / wg_dims.1,
                1,
            );
        }

        let index = app.queue.submit(std::iter::once(encoder.finish()));
        app.device.poll(wgpu::Maintain::WaitForSubmissionIndex(index)).panic_on_timeout();
    }
}
