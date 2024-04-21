use crate::{scene, state::RawState};

use super::texture_set::TextureSet;

use rtmath::basic_octree::*;

pub(super) struct PathTracer {
    uniform_buffer: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,

    texture_bgl: wgpu::BindGroupLayout,
    texture_bgs: Option<[wgpu::BindGroup; 2]>,

    scene_tree_buffer: wgpu::Buffer,
    scene_materials_buffer: wgpu::Buffer,
    scene_bg: wgpu::BindGroup,

    pipeline: wgpu::ComputePipeline,
}

impl PathTracer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> color_eyre::Result<PathTracer> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tracer.rs path tracer shader"),
            source: wgpu::ShaderSource::Wgsl(std::fs::read_to_string("shaders/out/path_tracer.wgsl")?.into()),
        });

        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tracer.rs path tracer uniform bind group layout"),
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

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tracer.rs path tracer uniform buffer"),
            size: std::mem::size_of::<RawState>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tracer.rs path tracer uniform bind group"),
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
            label: Some("tracer.rs path tracer textures bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: wgpu::TextureFormat::Rgba32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let file = std::fs::read("./vox/chr_knight.vox")?;
        let tree = vox_loader::File::from_bytes(file.as_slice())?;
        let voxels = tree
            .iter() //
            .filter_map(|v| v.chunk.ok().map(|c| (c, v.state)))
            .filter_map(|v| if let vox_loader::Chunk::Voxels(vox) = v.0 { Some((vox, v.1)) } else { None })
            .next()
            .unwrap();
        let voxel_count = voxels.1.size[0] as usize * voxels.1.size[1] as usize * voxels.1.size[2] as usize;

        let mut octree = rtmath::basic_octree::Octree::new(voxels.0.iter().map(|v| (cgmath::point3(v[0] as i64, v[1] as i64, v[2] as i64), v[3])).collect());
        octree.split_until(24);
        println!("{:?}", &octree);
        let threaded = octree.thread();
        //println!("{:#?}", &threaded);

        log::debug!("{:?}, {:?}, {}", voxels.1.origin, voxels.1.size, voxels.0.len());

        let scene_tree_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tracer.rs path tracer scene tree buffer"),
            size: (16 + threaded.len() * std::mem::size_of::<ThreadedOctreeNode>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // header
        {
            let dims_as_le_bytes = voxels.1.size.map(|v| v.to_le_bytes());

            #[rustfmt::skip]
            queue.write_buffer(&scene_tree_buffer, 0, &[
                dims_as_le_bytes[0][0], dims_as_le_bytes[0][1], dims_as_le_bytes[0][2], dims_as_le_bytes[0][3],
                dims_as_le_bytes[1][0], dims_as_le_bytes[1][1], dims_as_le_bytes[1][2], dims_as_le_bytes[1][3],
                dims_as_le_bytes[2][0], dims_as_le_bytes[2][1], dims_as_le_bytes[2][2], dims_as_le_bytes[2][3],
                0, 0, 0, 0, // padding
            ]);
        }

        queue.write_buffer(&scene_tree_buffer, 16, bytemuck::cast_slice(threaded.as_slice()));

        let scene_materials_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tracer.rs path tracer scene materials buffer"),
            size: (8 * octree.get_data().len()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        queue.write_buffer(
            &scene_materials_buffer,
            0,
            bytemuck::cast_slice(
                octree //
                    .get_data()
                    .iter()
                    .map(|_v| [0xC0C0C001u32, 0x00000000u32])
                    .collect::<Vec<_>>()
                    .as_slice(),
            ),
        );

        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tracer.rs path tracer scene bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tracer.rs path tracer scene bind group"),
            layout: &scene_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &scene_tree_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &scene_materials_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tracer.rs path tracer pipeline layout"),
            bind_group_layouts: &[&uniform_bgl, &texture_bgl, &scene_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tracer.rs path tracer pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
        });

        Ok(Self {
            uniform_bg,
            uniform_buffer,

            texture_bgl,
            texture_bgs: None,

            scene_tree_buffer,
            scene_materials_buffer,
            scene_bg,

            pipeline,
        })
    }

    pub fn resize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dimensions: (u32, u32), texture_set: &TextureSet) {
        self.texture_bgs = Some([
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tracer.rs path tracer textures bind group (even frames)"),
                layout: &self.texture_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_set.double_buffer[0].radiance.1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&texture_set.double_buffer[0].geometry.1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&texture_set.double_buffer[1].radiance.1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&texture_set.double_buffer[1].geometry.1),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tracer.rs path tracer textures bind group (odd frames)"),
                layout: &self.texture_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_set.double_buffer[1].radiance.1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&texture_set.double_buffer[1].geometry.1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&texture_set.double_buffer[0].radiance.1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&texture_set.double_buffer[0].geometry.1),
                    },
                ],
            }),
        ]);
    }

    pub fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, texture_set: &TextureSet, state: &RawState, workgroup_size: (u32, u32)) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tracer.rs path tracer command encoder"),
        });

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[*state]));

        {
            let texture_bgs = self.texture_bgs.as_ref().unwrap();

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tracer.rs path tracer compute pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.uniform_bg, &[]);
            pass.set_bind_group(1, if (state.frame_no % 2) == 0 { &texture_bgs[0] } else { &texture_bgs[1] }, &[]);
            pass.set_bind_group(2, &self.scene_bg, &[]);

            let wg_counts = ((state.dimensions[0] + workgroup_size.0 - 1) / workgroup_size.0, (state.dimensions[1] + workgroup_size.1 - 1) / workgroup_size.1);
            pass.dispatch_workgroups(wg_counts.0, wg_counts.1, 1);
        }

        device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(queue.submit(std::iter::once(encoder.finish()))));
    }
}
