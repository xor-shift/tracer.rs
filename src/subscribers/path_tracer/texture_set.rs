use super::geometry::GeometryElement;

pub struct TextureSet {
    extent: (u32, u32),
    extent_on_memory: (u32, u32),

    // input into path tracer
    pub pack_position_distance: (wgpu::Texture, wgpu::TextureView),
    pub object_indices: (wgpu::Texture, wgpu::TextureView),
    pub albedo: (wgpu::Texture, wgpu::TextureView),
    pub pack_normal_depth: (wgpu::Texture, wgpu::TextureView),
    pub geometry_pack_0: (wgpu::Texture, wgpu::TextureView),
    pub geometry_pack_1: (wgpu::Texture, wgpu::TextureView),

    // output from the path tracer
    //pub g_buffer: wgpu::Buffer,
    pub ray_trace_0: (wgpu::Texture, wgpu::TextureView),
    pub ray_trace_1: (wgpu::Texture, wgpu::TextureView),

    // output from the denoiser
    pub denoise_0: (wgpu::Texture, wgpu::TextureView),
    pub denoise_1: (wgpu::Texture, wgpu::TextureView),
}

impl TextureSet {
    pub fn make_bgle_storage_cs(binding: u32, rw: bool, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: if rw { wgpu::StorageTextureAccess::ReadWrite } else { wgpu::StorageTextureAccess::ReadOnly },
                format,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        }
    }

    pub fn make_bgle_regular_cs(binding: u32, is_float: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: if is_float { wgpu::TextureSampleType::Float { filterable: false } } else { wgpu::TextureSampleType::Uint },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }
    }

    pub fn make_bgle_regular_fs(binding: u32, is_float: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: if is_float { wgpu::TextureSampleType::Float { filterable: false } } else { wgpu::TextureSampleType::Uint },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }
    }

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
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };

        let generate_pair = |desc: &wgpu::TextureDescriptor| {
            let texture = device.create_texture(desc);
            let view = texture.create_view(&std::default::Default::default());
            (texture, view)
        };

        /*let g_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: extent_on_memory.0 as u64 * extent_on_memory.1 as u64 * std::mem::size_of::<GeometryElement>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });*/

        Self {
            extent,
            extent_on_memory,

            albedo: generate_pair(&wgpu::TextureDescriptor {
                label: Some("tracer.rs geometry texture (albedo and variance pack)"),
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

            geometry_pack_0: generate_pair(&wgpu::TextureDescriptor {
                label: Some("tracer.rs geometry texture (first pack)"),
                format: wgpu::TextureFormat::Rgba32Uint,
                ..texture_desc_geometry
            }),
            geometry_pack_1: generate_pair(&wgpu::TextureDescriptor {
                label: Some("tracer.rs geometry texture (second pack)"),
                format: wgpu::TextureFormat::Rgba32Uint,
                ..texture_desc_geometry
            }),

            ray_trace_0: generate_pair(&texture_desc_storage),
            ray_trace_1: generate_pair(&texture_desc_storage),
            denoise_0: generate_pair(&texture_desc_storage),
            denoise_1: generate_pair(&texture_desc_storage),
        }
    }
}
