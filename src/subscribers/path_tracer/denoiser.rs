use cgmath::num_traits::ToBytes;

use crate::Application;

use super::{state::State, texture_set::TextureSet};

pub enum DenoiserMode {
    IllumToD0,
    D0ToD1,
    D1ToD0,
}

pub struct Denoiser {
    config_buffer: wgpu::Buffer,
    bgl_config: wgpu::BindGroupLayout,
    bg_config: wgpu::BindGroup,

    bgl_textures: wgpu::BindGroupLayout,
    bg_pt_to_d0: wgpu::BindGroup,
    bg_pt_to_d0_swapped: wgpu::BindGroup,
    bg_d0_to_d1: wgpu::BindGroup,
    bg_d1_to_d0: wgpu::BindGroup,

    pipeline: wgpu::ComputePipeline,
}

impl Denoiser {
    fn make_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let desc = wgpu::BindGroupLayoutDescriptor {
            label: Some("tracer.rs denoiser texture set bind group layout"),
            entries: &[
                TextureSet::make_bgle_regular_cs(0, true),                                  // pt results or the previous denoiser output
                TextureSet::make_bgle_regular_cs(1, false),                                 // geometry pack 0
                TextureSet::make_bgle_regular_cs(2, false),                                 // geometry pack 1
                TextureSet::make_bgle_storage_cs(3, true, wgpu::TextureFormat::Rgba8Unorm), // denoiser output
            ],
        };

        device.create_bind_group_layout(&desc)
    }

    fn make_bg(texture_set: &TextureSet, device: &wgpu::Device, layout: &wgpu::BindGroupLayout, output: &wgpu::TextureView, input: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tracer.rs denoiser texture set bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input),
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
                    resource: wgpu::BindingResource::TextureView(&output),
                },
            ],
        })
    }

    pub fn new(app: &mut Application, texture_set: &TextureSet) -> color_eyre::Result<Denoiser> {
        let shader = app.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tracer.rs denoiser shader"),
            source: wgpu::ShaderSource::Wgsl(std::fs::read_to_string("./shaders/out/denoiser.wgsl")?.as_str().into()),
        });

        let config_buffer = app.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tracer.rs denoiser configuration buffer"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let bgl_config = app.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tracer.rs denoiser configuration bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bg_config = app.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tracer.rs denoiser configuration bind group"),
            layout: &bgl_config,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: config_buffer.as_entire_binding(),
            }],
        });

        let bgl_textures = Self::make_bgl(&app.device);
        let bg_pt_to_d0 = Self::make_bg(texture_set, &app.device, &bgl_textures, &texture_set.denoise_0.1, &texture_set.ray_trace_0.1);
        let bg_pt_to_d0_swapped = Self::make_bg(texture_set, &app.device, &bgl_textures, &texture_set.denoise_0.1, &texture_set.ray_trace_1.1);
        let bg_d0_to_d1 = Self::make_bg(texture_set, &app.device, &bgl_textures, &texture_set.denoise_1.1, &texture_set.denoise_0.1);
        let bg_d1_to_d0 = Self::make_bg(texture_set, &app.device, &bgl_textures, &texture_set.denoise_0.1, &texture_set.denoise_1.1);

        let pipeline_layout = app.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tracer.rs denoiser compute pipeline layout"),
            bind_group_layouts: &[&bgl_config, &bgl_textures],
            push_constant_ranges: &[],
        });

        let pipeline = app.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tracer.rs denoiser compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
        });

        Ok(Self {
            config_buffer,
            bgl_config,
            bg_config,

            bgl_textures,
            bg_pt_to_d0,
            bg_pt_to_d0_swapped,
            bg_d0_to_d1,
            bg_d1_to_d0,

            pipeline,
        })
    }

    pub fn resize(&mut self, app: &Application, texture_set: &TextureSet) {
        let bg_pt_to_d0 = Self::make_bg(texture_set, &app.device, &self.bgl_textures, &texture_set.denoise_0.1, &texture_set.ray_trace_0.1);
        let bg_pt_to_d0_swapped = Self::make_bg(texture_set, &app.device, &self.bgl_textures, &texture_set.denoise_0.1, &texture_set.ray_trace_1.1);
        let bg_d0_to_d1 = Self::make_bg(texture_set, &app.device, &self.bgl_textures, &texture_set.denoise_1.1, &texture_set.denoise_0.1);
        let bg_d1_to_d0 = Self::make_bg(texture_set, &app.device, &self.bgl_textures, &texture_set.denoise_0.1, &texture_set.denoise_1.1);

        self.bg_pt_to_d0 = bg_pt_to_d0;
        self.bg_pt_to_d0_swapped = bg_pt_to_d0_swapped;
        self.bg_d0_to_d1 = bg_d0_to_d1;
        self.bg_d1_to_d0 = bg_d1_to_d0;
    }

    pub fn render(&mut self, app: &mut Application, mode: DenoiserMode, stride: i32, state: &State) {
        app.queue.write_buffer(&self.config_buffer, 0, &stride.to_le_bytes());

        let mut encoder = app.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tracer.rs denoiser encoder") });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tracer.rs denoiser compute pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);

            let textures_bg = match mode {
                DenoiserMode::IllumToD0 => {
                    if state.should_swap_buffers() {
                        &self.bg_pt_to_d0_swapped
                    } else {
                        &self.bg_pt_to_d0
                    }
                }
                DenoiserMode::D0ToD1 => &self.bg_d0_to_d1,
                DenoiserMode::D1ToD0 => &self.bg_d1_to_d0,
            };
            compute_pass.set_bind_group(0, &self.bg_config, &[]);
            compute_pass.set_bind_group(1, textures_bg, &[]);

            let wg_dims = (8, 8);
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
