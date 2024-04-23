use crate::{scene, state::RawState};

use super::texture_set::TextureSet;

use rtmath::basic_octree::*;

pub(super) struct PathTracer {
    uniform_buffer: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,

    texture_bgl: wgpu::BindGroupLayout,
    texture_bgs: Option<[wgpu::BindGroup; 2]>,

    scene_tree_buffer: wgpu::Buffer,
    scene_bg: wgpu::BindGroup,

    pipeline: wgpu::ComputePipeline,
}

pub const fn get_test_tree() -> [NewThreadedOctreeNode; 80] {
    let sn = u32::MAX;

    let gray = [0x01333333, 0x00000000];
    let lgray = [0x01555555, 0x00000000];
    let red = [0x01FF0000, 0x00000000];
    let green = [0x0100FF00, 0x00000000];
    let blue = [0x010000FF, 0x00000000];

    #[rustfmt::skip]
    [
        NewThreadedOctreeNode::new_quick(&[],        [1, sn],  [0, 0]),
        NewThreadedOctreeNode::new_quick(&[0],       [2, 38],  [0, 0]),
        NewThreadedOctreeNode::new_quick(&[0, 0],    [3, 3],   gray),
        NewThreadedOctreeNode::new_quick(&[0, 1],    [4, 5],   [0, 0]),
        NewThreadedOctreeNode::new_quick(&[0, 1, 7], [5, 5],   red),
        NewThreadedOctreeNode::new_quick(&[0, 2],    [6, 7],   [0, 0]),
        NewThreadedOctreeNode::new_quick(&[0, 2, 7], [7, 7],   red),
        NewThreadedOctreeNode::new_quick(&[0, 3],    [8, 13],  [0, 0]),
        NewThreadedOctreeNode::new_quick(&[0, 3, 1], [9, 9],   red),
        NewThreadedOctreeNode::new_quick(&[0, 3, 2], [10, 10], red),
        NewThreadedOctreeNode::new_quick(&[0, 3, 4], [11, 11], red),
        NewThreadedOctreeNode::new_quick(&[0, 3, 5], [12, 12], red),
        NewThreadedOctreeNode::new_quick(&[0, 3, 6], [13, 13], red),
        NewThreadedOctreeNode::new_quick(&[0, 4],    [14, 15], [0, 0]),
        NewThreadedOctreeNode::new_quick(&[0, 4, 7], [15, 15], red),
        NewThreadedOctreeNode::new_quick(&[0, 5],    [16, 23], [0, 0]),
        NewThreadedOctreeNode::new_quick(&[0, 5, 1], [17, 17], red),
        NewThreadedOctreeNode::new_quick(&[0, 5, 2], [18, 18], red),
        NewThreadedOctreeNode::new_quick(&[0, 5, 3], [19, 19], red),
        NewThreadedOctreeNode::new_quick(&[0, 5, 4], [20, 20], red),
        NewThreadedOctreeNode::new_quick(&[0, 5, 5], [21, 21], red),
        NewThreadedOctreeNode::new_quick(&[0, 5, 6], [22, 22], red),
        NewThreadedOctreeNode::new_quick(&[0, 5, 7], [23, 23], red),
        NewThreadedOctreeNode::new_quick(&[0, 6],    [24, 31], [0, 0]),
        NewThreadedOctreeNode::new_quick(&[0, 6, 1], [25, 25], red),
        NewThreadedOctreeNode::new_quick(&[0, 6, 2], [26, 26], red),
        NewThreadedOctreeNode::new_quick(&[0, 6, 3], [27, 27], red),
        NewThreadedOctreeNode::new_quick(&[0, 6, 4], [28, 28], red),
        NewThreadedOctreeNode::new_quick(&[0, 6, 5], [29, 29], red),
        NewThreadedOctreeNode::new_quick(&[0, 6, 6], [30, 30], red),
        NewThreadedOctreeNode::new_quick(&[0, 6, 7], [31, 31], red),
        NewThreadedOctreeNode::new_quick(&[0, 7],    [32, 38], [0, 0]),
        NewThreadedOctreeNode::new_quick(&[0, 7, 0], [33, 33], red),
        NewThreadedOctreeNode::new_quick(&[0, 7, 1], [34, 34], red),
        NewThreadedOctreeNode::new_quick(&[0, 7, 2], [35, 35], red),
        NewThreadedOctreeNode::new_quick(&[0, 7, 4], [36, 36], red),
        NewThreadedOctreeNode::new_quick(&[0, 7, 5], [37, 37], red),
        NewThreadedOctreeNode::new_quick(&[0, 7, 6], [38, 38], red),
        NewThreadedOctreeNode::new_quick(&[1],       [39, 74], [0, 0]),
        NewThreadedOctreeNode::new_quick(&[1, 0],    [40, 41], [0, 0]),
        NewThreadedOctreeNode::new_quick(&[1, 0, 6], [41, 41], green),
        NewThreadedOctreeNode::new_quick(&[1, 2],    [42, 47], [0, 0]),
        NewThreadedOctreeNode::new_quick(&[1, 2, 0], [43, 43], green),
        NewThreadedOctreeNode::new_quick(&[1, 2, 3], [44, 44], green),
        NewThreadedOctreeNode::new_quick(&[1, 2, 4], [45, 45], green),
        NewThreadedOctreeNode::new_quick(&[1, 2, 5], [46, 46], green),
        NewThreadedOctreeNode::new_quick(&[1, 2, 7], [47, 47], green),
        NewThreadedOctreeNode::new_quick(&[1, 3],    [48, 49], [0, 0]),
        NewThreadedOctreeNode::new_quick(&[1, 3, 6], [49, 49], green),
        NewThreadedOctreeNode::new_quick(&[1, 4],    [50, 57], [0, 0]),
        NewThreadedOctreeNode::new_quick(&[1, 4, 0], [51, 51], green),
        NewThreadedOctreeNode::new_quick(&[1, 4, 2], [52, 52], green),
        NewThreadedOctreeNode::new_quick(&[1, 4, 3], [53, 53], green),
        NewThreadedOctreeNode::new_quick(&[1, 4, 4], [54, 54], green),
        NewThreadedOctreeNode::new_quick(&[1, 4, 5], [55, 55], green),
        NewThreadedOctreeNode::new_quick(&[1, 4, 6], [56, 56], green),
        NewThreadedOctreeNode::new_quick(&[1, 4, 7], [57, 57], green),
        NewThreadedOctreeNode::new_quick(&[1, 5],    [58, 59], [0, 0]),
        NewThreadedOctreeNode::new_quick(&[1, 5, 6], [59, 59], green),
        NewThreadedOctreeNode::new_quick(&[1, 6],    [60, 66], [0, 0]),
        NewThreadedOctreeNode::new_quick(&[1, 6, 0], [61, 61], green),
        NewThreadedOctreeNode::new_quick(&[1, 6, 1], [62, 62], green),
        NewThreadedOctreeNode::new_quick(&[1, 6, 3], [63, 63], green),
        NewThreadedOctreeNode::new_quick(&[1, 6, 4], [64, 64], green),
        NewThreadedOctreeNode::new_quick(&[1, 6, 5], [65, 65], green),
        NewThreadedOctreeNode::new_quick(&[1, 6, 7], [66, 66], green),
        NewThreadedOctreeNode::new_quick(&[1, 7],    [67, 74], [0, 0]),
        NewThreadedOctreeNode::new_quick(&[1, 7, 0], [68, 68], green),
        NewThreadedOctreeNode::new_quick(&[1, 7, 2], [69, 69], green),
        NewThreadedOctreeNode::new_quick(&[1, 7, 3], [70, 70], green),
        NewThreadedOctreeNode::new_quick(&[1, 7, 4], [71, 71], green),
        NewThreadedOctreeNode::new_quick(&[1, 7, 5], [72, 72], green),
        NewThreadedOctreeNode::new_quick(&[1, 7, 6], [73, 73], green),
        NewThreadedOctreeNode::new_quick(&[1, 7, 7], [74, 74], green),
        NewThreadedOctreeNode::new_quick(&[2],       [75, 75], blue),
        NewThreadedOctreeNode::new_quick(&[3],       [76, 76], red),
        NewThreadedOctreeNode::new_quick(&[4],       [77, 77], green),
        NewThreadedOctreeNode::new_quick(&[5],       [78, 78], blue),
        NewThreadedOctreeNode::new_quick(&[6],       [79, 79], red),
        NewThreadedOctreeNode::new_quick(&[7],       [sn, sn], green        ),
    ]
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

        /*let file = std::fs::read("./vox/chr_knight.vox")?;
        let tree = vox_loader::File::from_bytes(file.as_slice())?;
        let voxels = tree
            .iter() //
            .filter_map(|v| v.chunk.ok().map(|c| (c, v.state)))
            .filter_map(|v| if let vox_loader::Chunk::Voxels(vox) = v.0 { Some((vox, v.1)) } else { None })
            .next()
            .unwrap();
        let voxel_count = voxels.1.size[0] as usize * voxels.1.size[1] as usize * voxels.1.size[2] as usize;

        let mut octree = rtmath::basic_octree::Octree::new(voxels.0.iter().map(|v| (cgmath::point3(v[0] as i64, v[1] as i64, v[2] as i64), v[3])).collect());
        octree.split_until(16);
        println!("{:?}", &octree);
        let threaded = octree.thread();
        //println!("{:#?}", &threaded);

        log::debug!("{:?}, {:?}, {}", voxels.1.origin, voxels.1.size, voxels.0.len());*/

        let threaded_tree = get_test_tree();

        let scene_tree_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tracer.rs path tracer scene tree buffer"),
            size: (16 + threaded_tree.len() * std::mem::size_of::<ThreadedOctreeNode>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // header
        {
            //let dims_as_le_bytes = voxels.1.size.map(|v| v.to_le_bytes());
            let dims_as_le_bytes = [[8, 0, 0, 0], [8, 0, 0, 0], [8, 0, 0, 0]];

            #[rustfmt::skip]
            queue.write_buffer(&scene_tree_buffer, 0, &[
                dims_as_le_bytes[0][0], dims_as_le_bytes[0][1], dims_as_le_bytes[0][2], dims_as_le_bytes[0][3],
                dims_as_le_bytes[1][0], dims_as_le_bytes[1][1], dims_as_le_bytes[1][2], dims_as_le_bytes[1][3],
                dims_as_le_bytes[2][0], dims_as_le_bytes[2][1], dims_as_le_bytes[2][2], dims_as_le_bytes[2][3],
                0, 0, 0, 0, // padding
            ]);
        }

        //queue.write_buffer(&scene_tree_buffer, 16, bytemuck::cast_slice(threaded.as_slice()));

        queue.write_buffer(&scene_tree_buffer, 16, bytemuck::cast_slice(&threaded_tree));

        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tracer.rs path tracer scene bind group layout"),
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

        let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tracer.rs path tracer scene bind group"),
            layout: &scene_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &scene_tree_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
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
