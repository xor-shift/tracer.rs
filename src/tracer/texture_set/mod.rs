pub struct TextureAndView(pub wgpu::Texture, pub wgpu::TextureView);

impl TextureAndView {
    pub fn new(device: &wgpu::Device, dimensions: (u32, u32), format: wgpu::TextureFormat, usage: wgpu::TextureUsages, array_depth: u32) -> TextureAndView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: dimensions.0,
                height: dimensions.1,
                depth_or_array_layers: array_depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });

        let view = if array_depth == 1 {
            texture.create_view(&std::default::Default::default())
        } else {
            texture.create_view(&wgpu::TextureViewDescriptor {
                base_array_layer: 0,
                array_layer_count: Some(2),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..std::default::Default::default()
            })
        };

        Self(texture, view)
    }
}

pub(super) struct SingleTextureSet {
    pub radiance: TextureAndView,
    pub geometry: TextureAndView,
}

impl SingleTextureSet {
    pub fn new(device: &wgpu::Device, dimensions: (u32, u32)) -> SingleTextureSet {
        Self {
            radiance: TextureAndView::new(device, dimensions, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING, 1),
            geometry: TextureAndView::new(device, dimensions, wgpu::TextureFormat::Rgba32Uint, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING, 2),
        }
    }
}

pub(super) struct TextureSet {
    pub double_buffer: [SingleTextureSet; 2],
    pub denoise_buffers: TextureAndView,
}

impl TextureSet {
    pub fn new(device: &wgpu::Device, dimensions: (u32, u32)) -> TextureSet {
        Self {
            double_buffer: [SingleTextureSet::new(device, dimensions), SingleTextureSet::new(device, dimensions)],
            denoise_buffers: TextureAndView::new(device, dimensions, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING, 2),
        }
    }
}

pub mod quick_bgl_entries {}
