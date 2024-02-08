use super::geometry::GeometryElement;

pub struct TextureSet {
    extent: (u32, u32),
    extent_on_memory: (u32, u32),
    ray_trace: (wgpu::Texture, wgpu::TextureView),
    denoise_0: (wgpu::Texture, wgpu::TextureView),
    denoise_1: (wgpu::Texture, wgpu::TextureView),
    g_buffer: wgpu::Buffer,
}

impl TextureSet {
    pub fn new(extent: (u32, u32), device: &wgpu::Device) -> Self {
        let extent_on_memory = (((extent.0 + 7) / 8) * 8, ((extent.1 + 7) / 8) * 8);

        let texture_desc = wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: extent_on_memory.0,
                height: extent_on_memory.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };

        let generate_pair = || {
            let texture = device.create_texture(&texture_desc);
            let view = texture.create_view(&std::default::Default::default());
            (texture, view)
        };

        let g_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: extent_on_memory.0 as u64 * extent_on_memory.1 as u64 * std::mem::size_of::<GeometryElement>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Self {
            extent,
            extent_on_memory,
            ray_trace: generate_pair(),
            denoise_0: generate_pair(),
            denoise_1: generate_pair(),
            g_buffer,
        }
    }

    pub fn bind_group_layout_cs(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let default_tex_bind = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
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
                wgpu::BindGroupLayoutEntry { binding: 0, ..default_tex_bind }, //
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { binding: 2, ..default_tex_bind },
                wgpu::BindGroupLayoutEntry { binding: 3, ..default_tex_bind },
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
                wgpu::BindGroupLayoutEntry { binding: 0, ..tex_bind }, //
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { binding: 2, ..storage_tex_bind },
                wgpu::BindGroupLayoutEntry { binding: 3, ..storage_tex_bind },
            ],
        };

        device.create_bind_group_layout(&desc)
    }

    pub fn bind_group_cs(&self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::BindGroup {
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
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.g_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.denoise_0.1),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.denoise_1.1),
                },
            ],
        })
    }

    pub fn bind_group_fs(&self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::BindGroup {
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
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.g_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.denoise_0.1),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.denoise_1.1),
                },
            ],
        })
    }
}
