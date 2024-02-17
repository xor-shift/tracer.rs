#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 4],
    pub normal: [f32; 4],
    pub material: u32,
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
            0 => Float32x4,
            1 => Float32x4,
            2 => Uint32,
        ];

        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Triangle {
    pub vertices: [[f32; 4]; 3],
    pub material: u32,
    pub padding_0: u32,
    pub padding_1: u32,
    pub padding_2: u32,
}

impl Triangle {
    pub const fn new(vertices: [[f32; 3]; 3], material: u32) -> Triangle {
        Self {
            vertices: [
                [vertices[0][0], vertices[0][1], vertices[0][2], 1.],
                [vertices[1][0], vertices[1][1], vertices[1][2], 1.],
                [vertices[2][0], vertices[2][1], vertices[2][2], 1.],
            ],
            material,
            padding_0: 0,
            padding_1: 0,
            padding_2: 0,
        }
    }

    pub fn into_vertices(&self) -> [Vertex; 3] {
        let e0 = [
            self.vertices[1][0] - self.vertices[0][0], //
            self.vertices[1][1] - self.vertices[0][1], //
            self.vertices[1][2] - self.vertices[0][2], //
        ];

        let e1 = [
            self.vertices[2][0] - self.vertices[0][0], //
            self.vertices[2][1] - self.vertices[0][1], //
            self.vertices[2][2] - self.vertices[0][2], //
        ];

        let normal = [
            (e0[1] * e1[2]) - (e0[2] * e1[1]), //
            (e0[2] * e1[0]) - (e0[0] * e1[2]), //
            (e0[0] * e1[1]) - (e0[1] * e1[0]), //
        ];

        let normal_len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();

        let normal = [normal[0] / normal_len, normal[1] / normal_len, normal[2] / normal_len, 1.];

        [
            Vertex {
                position: self.vertices[0],
                normal,
                material: self.material,
            },
            Vertex {
                position: self.vertices[1],
                normal,
                material: self.material,
            },
            Vertex {
                position: self.vertices[2],
                normal,
                material: self.material,
            },
        ]
    }
}

pub fn triangles_into_vertices<const N: usize>(triangles: &[Triangle; N]) -> [Vertex; N * 3] {
    let mut ret = [Vertex {
        position: [0.; 4],
        normal: [0.; 4],
        material: 0,
    }; N * 3];

    let mut i = 0;
    loop {
        if i >= N {
            break;
        }

        let tmp = triangles[i].into_vertices();
        ret[i * 3 + 0] = tmp[0];
        ret[i * 3 + 1] = tmp[1];
        ret[i * 3 + 2] = tmp[2];

        i += 1;
    }

    ret
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Voxel {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub material: u32,
}

impl Voxel {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Uint32,
        ];

        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRS,
        }
    }
}
