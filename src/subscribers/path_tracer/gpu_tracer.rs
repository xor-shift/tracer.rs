use super::noise_texture::NoiseTexture;
use super::texture_set::TextureSet;
use super::uniform::RawMainUniform;
use super::uniform::UniformGenerator;

use crate::subscriber::*;
use crate::Application;

use wgpu::util::DeviceExt;
use wgpu::BindGroupDescriptor;

pub struct GPUTracer {
    pub uniform_generator: UniformGenerator,
    uniform_buffer: wgpu::Buffer,
    bind_group_main: wgpu::BindGroup,

    pub texture_set_bgl: wgpu::BindGroupLayout,
    pub texture_set_bg: wgpu::BindGroup,

    triangles_buffer: wgpu::Buffer,
    triangles_bgl: wgpu::BindGroupLayout,
    triangles_bg: wgpu::BindGroup,

    pipeline: wgpu::ComputePipeline,
}

impl GPUTracer {
    pub fn new(app: &mut Application, texture_set: &TextureSet, g_buffer: &wgpu::Buffer) -> color_eyre::Result<GPUTracer> {
        let extent: (u32, u32) = app.window.inner_size().into();

        let shader = app.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tracer.rs rendering shader"),
            source: wgpu::ShaderSource::Wgsl(std::fs::read_to_string("./shaders/out/compute.wgsl")?.as_str().into()),
        });

        let mut uniform_generator = UniformGenerator::new(app.window.inner_size().into());
        let uniform_buffer: wgpu::Buffer = app.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("main uniform"),
            contents: bytemuck::cast_slice(&[std::convert::Into::<RawMainUniform>::into(uniform_generator.generate())]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&noise.view),
                },
            ],
        });

        let texture_set_bgl = TextureSet::bind_group_layout_cs(&app.device);
        let texture_set_bg = texture_set.bind_group_cs(&app.device, &texture_set_bgl, texture_set, g_buffer);

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
            label: Some("tracer.rs main pipeline layout"),
            bind_group_layouts: &[&bind_group_layout_main, &texture_set_bgl, &triangles_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = app.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tracer.rs main compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
        });

        Ok(Self {
            uniform_generator,

            uniform_buffer,
            bind_group_main,

            texture_set_bgl,
            texture_set_bg,

            triangles_buffer,
            triangles_bgl,
            triangles_bg,

            pipeline,
        })
    }

    pub fn resize(&mut self, app: &mut Application, new_size: winit::dpi::PhysicalSize<u32>, texture_set: &TextureSet, g_buffer: &wgpu::Buffer) {
        self.uniform_generator.dimensions = new_size.into();
        self.uniform_generator.reset();

        let texture_set_bg = texture_set.bind_group_cs(&app.device, &self.texture_set_bgl, texture_set, g_buffer);
        self.texture_set_bg = texture_set_bg;
    }

    pub fn render(&mut self, app: &mut Application, camera_position: [f32; 3]) {
        let raw_uniform = RawMainUniform {
            camera_position,
            ..self.uniform_generator.frame_start()
        };
        app.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[raw_uniform]));

        let mut encoder = app.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tracer.rs rasterisation encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tracer.rs compute pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group_main, &[]);
            compute_pass.set_bind_group(1, &self.texture_set_bg, &[]);
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

        self.uniform_generator.frame_end();
    }
}
