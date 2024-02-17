use super::geometry::GeometryElement;

pub struct TextureSet {
    extent: (u32, u32),
    extent_on_memory: (u32, u32),
    ray_trace: (wgpu::Texture, wgpu::TextureView),
    denoise_0: (wgpu::Texture, wgpu::TextureView),
    denoise_1: (wgpu::Texture, wgpu::TextureView),

    pub pack_position_distance: (wgpu::Texture, wgpu::TextureView),
    pub object_indices: (wgpu::Texture, wgpu::TextureView),
    pub albedo: (wgpu::Texture, wgpu::TextureView),
    pub pack_normal_depth: (wgpu::Texture, wgpu::TextureView),
}

impl TextureSet {
    pub fn new(extent: (u32, u32), device: &wgpu::Device) -> Self {
        let extent_on_memory = (((extent.0 + 7) / 8) * 8, ((extent.1 + 7) / 8) * 8);

        let texture_desc_storage = wgpu::TextureDescriptor {
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
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };

        let texture_desc_geometry = wgpu::TextureDescriptor {
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
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };

        let generate_pair = |desc: &wgpu::TextureDescriptor| {
            let texture = device.create_texture(desc);
            let view = texture.create_view(&std::default::Default::default());
            (texture, view)
        };

        Self {
            extent,
            extent_on_memory,
            ray_trace: generate_pair(&texture_desc_storage),
            denoise_0: generate_pair(&texture_desc_storage),
            denoise_1: generate_pair(&texture_desc_storage),

            albedo: generate_pair(&wgpu::TextureDescriptor {
                label: Some("tracer.rs geometry texture (albedo and triangle ID pack)"),
                format: wgpu::TextureFormat::Rgba16Float,
                ..texture_desc_geometry
            }),

            pack_normal_depth: generate_pair(&wgpu::TextureDescriptor {
                label: Some("tracer.rs geometry texture (normal and depth pack)"),
                format: wgpu::TextureFormat::Rgba32Float,
                ..texture_desc_geometry
            }),

            pack_position_distance: generate_pair(&wgpu::TextureDescriptor {
                label: Some("tracer.rs geometry texture (position and distance pack)"),
                format: wgpu::TextureFormat::Rgba32Float,
                ..texture_desc_geometry
            }),

            object_indices: generate_pair(&wgpu::TextureDescriptor {
                label: Some("tracer.rs geometry texture (first-hit geometry indices)"),
                format: wgpu::TextureFormat::R32Uint,
                ..texture_desc_geometry
            }),
        }
    }

    pub fn bind_group_layout_cs(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let storage_tex_bind = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::ReadWrite,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };

        let tex_bind = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        let desc = wgpu::BindGroupLayoutDescriptor {
            label: Some("tracer.rs pt texture set bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, ..storage_tex_bind }, //
                wgpu::BindGroupLayoutEntry { binding: 1, ..tex_bind },         // albedo
                wgpu::BindGroupLayoutEntry { binding: 2, ..tex_bind },         // pack of normal and depth
                wgpu::BindGroupLayoutEntry { binding: 3, ..tex_bind },         // pack of position and distance
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    ..tex_bind
                }, // object index
                wgpu::BindGroupLayoutEntry { binding: 5, ..storage_tex_bind },
                wgpu::BindGroupLayoutEntry { binding: 6, ..storage_tex_bind },
            ],
        };

        device.create_bind_group_layout(&desc)
    }

    pub fn bind_group_layout_fs(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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

        let storage_tex_bind = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::ReadWrite,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };

        let desc = wgpu::BindGroupLayoutDescriptor {
            label: Some("tracer.rs pt texture set bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, ..tex_bind }, // path trace result
                wgpu::BindGroupLayoutEntry { binding: 1, ..tex_bind }, // albedo
                wgpu::BindGroupLayoutEntry { binding: 2, ..tex_bind }, // pack of normal and depth
                wgpu::BindGroupLayoutEntry { binding: 3, ..tex_bind }, // pack of position and distance
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    ..tex_bind
                }, // object index
                wgpu::BindGroupLayoutEntry { binding: 5, ..tex_bind }, // denoise_0
                wgpu::BindGroupLayoutEntry { binding: 6, ..tex_bind }, // denoise_1
            ],
        };

        device.create_bind_group_layout(&desc)
    }

    pub fn bind_group_cs(&self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout, texture_set: &TextureSet, g_buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tracer.rs pt texture set bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.ray_trace.1),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_set.albedo.1),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&texture_set.pack_normal_depth.1),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&texture_set.pack_position_distance.1),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&texture_set.object_indices.1),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.denoise_0.1),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.denoise_1.1),
                },
            ],
        })
    }

    pub fn bind_group_fs(&self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout, texture_set: &TextureSet, g_buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tracer.rs pt texture set bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.ray_trace.1),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_set.albedo.1),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&texture_set.pack_normal_depth.1),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&texture_set.pack_position_distance.1),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&texture_set.object_indices.1),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.denoise_0.1),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.denoise_1.1),
                },
            ],
        })
    }
}
