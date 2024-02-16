#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub material: u32,
}

impl Vertex {
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

pub struct Triangle {
    pub vertices: [[f32; 3]; 3],
    pub material: u32,
}

impl Triangle {
    pub const fn into_vertices(&self) -> [Vertex; 3] {
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

pub const fn triangles_into_vertices<const N: usize>(triangles: &[Triangle; N]) -> [Vertex; N * 3] {
    let mut ret = [Vertex {
        position: [0.; 3],
        normal: [0.; 3],
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
