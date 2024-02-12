use stuff::rng::{RandomNumberEngine, UniformRandomBitGenerator};

pub struct NoiseTexture {
    pub actual_noise: Box<[u8]>,
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl NoiseTexture {
    pub fn new(dimensions: (u32, u32), device: &wgpu::Device, queue: &wgpu::Queue, label: Option<&str>) -> Self {
        let num_elems = dimensions.0 as usize * dimensions.1 as usize * 4;
        let num_bytes = num_elems * 4;
        //let mut data: Vec<_> = (0..num_elems).map(|_| 0).collect();
        let mut data = Box::new_uninit_slice(num_bytes);

        let mut rd = stuff::rng::engines::RandomDevice::new();
        let mut gen = stuff::rng::engines::Xoshiro256PP::new();
        gen.seed_from_result(rd.generate());
        drop(rd);

        for i in 0..num_bytes / 8 {
            let bytes: [_; 8] = unsafe { std::mem::transmute(gen.generate()) };
            for j in 0..8 {
                data[i * 8 + j].write(bytes[j]);
            }
        }

        for (i, v) in unsafe { std::mem::transmute::<_, [_; 8]>(gen.generate()) }.into_iter().enumerate().take(num_bytes % 8) {
            data[num_bytes - (num_bytes % 8) + i].write(v);
        }

        let data = unsafe { data.assume_init() };

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label.unwrap_or("random texture")),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            data.as_ref(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0 * 4),
                rows_per_image: Some(dimensions.1),
            },
            size,
        );

        let view = texture.create_view(&Default::default());

        Self { actual_noise: data, texture, view }
    }
}
