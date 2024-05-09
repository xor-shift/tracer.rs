use cgmath::InnerSpace;
use cgmath::MetricSpace;
use stuff::rng::distributions::GenerateCanonical;
use stuff::rng::RandomNumberEngine;
use stuff::rng::UniformRandomBitGenerator;

use crate::basic_octree::*;
use crate::state::RawState;

use super::texture_set::TextureSet;

pub(super) struct PathTracer {
    uniform_buffer: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,

    texture_bgl: wgpu::BindGroupLayout,
    texture_bgs: Option<[wgpu::BindGroup; 2]>,

    scene_bgl: wgpu::BindGroupLayout,
    scene_chunks_buffer: wgpu::Buffer,
    scene_node_pool_buffer: wgpu::Buffer,
    scene_bg: wgpu::BindGroup,

    pipeline: wgpu::ComputePipeline,
}

fn get_scene_tree(fast_decision: bool, spherical: bool) -> Vec<ThreadedOctreeNode> {
    let mut tree = VoxelTree::new([64; 3]);
    let mut rd = stuff::rng::engines::RandomDevice::new();
    let mut gen = stuff::rng::engines::Xoshiro256PP::new();
    gen.seed_from_result(rd.generate());
    drop(rd);

    let mut ct = 0;

    let tp_start = std::time::Instant::now();
    /*for x in 0..64 {
        for z in 0..64 {
            let height = (f64::generate_canonical(&mut gen) * 8. + 32.).floor() as u32;

            for y in 0..height {
                tree.set_voxel((x, y, z).into(), [0x01555555, 0x00000000]).unwrap();
                ct += 1;
            }
        }
    }*/

    for z in 0..64 {
        for y in 0..64 {
            for x in 0..64 {
                let vec = cgmath::vec3(x, y, z) //
                    .cast::<f32>()
                    .unwrap()
                    - cgmath::vec3(32., 32., 32.);

                let len = vec.dot(vec).sqrt();

                if spherical && len >= 20. {
                    continue;
                }

                let vec = vec / len;

                let decision = if fast_decision { ChildSelectionOrder::decide_fast(vec) } else { ChildSelectionOrder::decide_order(vec) };

                let colors = [
                    [127, 0, 0],
                    [0, 127, 0],
                    [127, 127, 0],
                    [0, 0, 127],
                    [127, 0, 127],
                    [0, 127, 127],
                    [127, 127, 127],
                    [255, 0, 0],
                    [0, 255, 0],
                    [255, 255, 0],
                    [0, 0, 255],
                    [255, 0, 255],
                    [0, 255, 255],
                    [255, 255, 255],
                ];

                let color = colors[decision];
                let material = [
                    0x01000000 | (color[0] << 16) | (color[1] << 8) | color[2], //
                    0x00000000,
                ];

                tree.set_voxel((x, y, z).into(), material).unwrap();
                ct += 1;
            }
        }
    }

    let tp_end = std::time::Instant::now();
    log::debug!("set {} voxels in {}s; tree has {} nodes and has a depth of {}", ct, (tp_end - tp_start).as_secs_f64(), tree.len(), tree.depth());

    //

    let tp_start = std::time::Instant::now();
    let threaded_tree = tree.thread_with_order([7, 6, 3, 5, 2, 1, 4, 0].into());
    let tp_end = std::time::Instant::now();

    log::debug!("produced {} threaded nodes in {}s", threaded_tree.len(), (tp_end - tp_start).as_secs_f64());

    threaded_tree
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct ChunkInformation {
    location: [i32; 3],
    pool_pointer: u32,
    size: [u32; 3],
    node_count: u32,
}

impl PathTracer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> color_eyre::Result<PathTracer> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tracer.rs path tracer shader"),
            source: wgpu::ShaderSource::Wgsl(std::fs::read_to_string("shaders/path_tracer.wgsl")?.into()),
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

        let tree_0 = get_scene_tree(false, false);
        let tree_1 = get_scene_tree(true, false);
        let tree_2 = get_scene_tree(false, true);
        let tree_3 = get_scene_tree(true, true);

        let scene_chunks_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tracer.rs path tracer scene tree buffer"),
            size: (4 * std::mem::size_of::<ChunkInformation>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &scene_chunks_buffer,
            0,
            bytemuck::cast_slice(&[
                ChunkInformation {
                    // slow, cube
                    location: [-32, 0, -32],
                    pool_pointer: 0,
                    size: [32; 3],
                    node_count: tree_0.len() as u32,
                },
                ChunkInformation {
                    // fast, cube
                    location: [0, 0, 0],
                    pool_pointer: tree_0.len() as u32,
                    size: [32; 3],
                    node_count: tree_1.len() as u32,
                },
                ChunkInformation {
                    // slow, sphere
                    location: [0, 0, -32],
                    pool_pointer: (tree_0.len() + tree_1.len()) as u32,
                    size: [32; 3],
                    node_count: tree_2.len() as u32,
                },
                ChunkInformation {
                    // fast, sphere
                    location: [-32, 0, 0],
                    pool_pointer: (tree_0.len() + tree_1.len() + tree_2.len()) as u32,
                    size: [32; 3],
                    node_count: tree_3.len() as u32,
                },
            ]),
        );

        let scene_node_pool_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tracer.rs path tracer node pool buffer"),
            size: ((tree_0.len() + tree_1.len() + tree_2.len() + tree_3.len()) * std::mem::size_of::<ThreadedOctreeNode>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        /*#[rustfmt::skip]
        {
            queue.write_buffer(&scene_node_pool_buffer, (threaded_tree.len() * std::mem::size_of::<ThreadedOctreeNode>() * 0) as wgpu::BufferAddress, bytemuck::cast_slice(&threaded_tree));
            queue.write_buffer(&scene_node_pool_buffer, (threaded_tree.len() * std::mem::size_of::<ThreadedOctreeNode>() * 1) as wgpu::BufferAddress, bytemuck::cast_slice(&threaded_tree));
            queue.write_buffer(&scene_node_pool_buffer, (threaded_tree.len() * std::mem::size_of::<ThreadedOctreeNode>() * 2) as wgpu::BufferAddress, bytemuck::cast_slice(&threaded_tree));
            queue.write_buffer(&scene_node_pool_buffer, (threaded_tree.len() * std::mem::size_of::<ThreadedOctreeNode>() * 3) as wgpu::BufferAddress, bytemuck::cast_slice(&threaded_tree));
            queue.write_buffer(&scene_node_pool_buffer, (threaded_tree.len() * std::mem::size_of::<ThreadedOctreeNode>() * 4) as wgpu::BufferAddress, bytemuck::cast_slice(&threaded_tree));
            queue.write_buffer(&scene_node_pool_buffer, (threaded_tree.len() * std::mem::size_of::<ThreadedOctreeNode>() * 5) as wgpu::BufferAddress, bytemuck::cast_slice(&threaded_tree));
        }*/

        #[rustfmt::skip]
        {
            queue.write_buffer(&scene_node_pool_buffer, 0, bytemuck::cast_slice(&tree_0));
            queue.write_buffer(&scene_node_pool_buffer, (tree_0.len() * std::mem::size_of::<ThreadedOctreeNode>()) as wgpu::BufferAddress, bytemuck::cast_slice(&tree_1));
            queue.write_buffer(&scene_node_pool_buffer, ((tree_0.len() + tree_1.len()) * std::mem::size_of::<ThreadedOctreeNode>()) as wgpu::BufferAddress, bytemuck::cast_slice(&tree_2));
            queue.write_buffer(&scene_node_pool_buffer, ((tree_0.len() + tree_1.len() + tree_2.len()) * std::mem::size_of::<ThreadedOctreeNode>()) as wgpu::BufferAddress, bytemuck::cast_slice(&tree_3));
        }

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
                        buffer: &scene_chunks_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &scene_node_pool_buffer,
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

            scene_bgl,
            scene_chunks_buffer,
            scene_node_pool_buffer,
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
