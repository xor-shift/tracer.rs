pub struct FontStuff {
    sampler: wgpu::Sampler,
    pub texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    pub bind_group: wgpu::BindGroup,
}

impl FontStuff {
    pub fn new(font_atlas: &mut imgui::FontAtlas, bgl: &wgpu::BindGroupLayout, device: &wgpu::Device, queue: &wgpu::Queue) -> FontStuff {
        let atlas_texture = font_atlas.build_rgba32_texture();

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tracer.rs imgui font atlas sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.,
            lod_max_clamp: 0.,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tracer.rs imgui font atlas texture"),
            size: wgpu::Extent3d {
                width: atlas_texture.width,
                height: atlas_texture.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        });

        queue.write_texture(
            wgpu::ImageCopyTextureBase {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            atlas_texture.data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * atlas_texture.width),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: atlas_texture.width,
                height: atlas_texture.height,
                depth_or_array_layers: 1,
            },
        );

        let texture_view = texture.create_view(&std::default::Default::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tracer.rs imgui font atlas texture bind group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
            ],
        });

        Self { sampler, texture, texture_view, bind_group }
    }
}
