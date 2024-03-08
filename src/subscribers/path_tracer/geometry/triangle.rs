use super::Vertex;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Triangle {
    pub vertex_0: [f32; 3],
    pub material: u32,
    pub vertex_1: [f32; 3],
    padding_0: u32,
    pub vertex_2: [f32; 3],
    padding_1: u32,
}

impl std::ops::Index<usize> for Triangle {
    type Output = [f32; 3];

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.vertex_0,
            1 => &self.vertex_1,
            2 => &self.vertex_2,
            _ => panic!("index out of bounds"),
        }
    }
}

impl std::ops::IndexMut<usize> for Triangle {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.vertex_0,
            1 => &mut self.vertex_1,
            2 => &mut self.vertex_2,
            _ => panic!("index out of bounds"),
        }
    }
}

impl Triangle {
    pub const fn new(vertices: [[f32; 3]; 3], material: u32) -> Triangle {
        Self {
            vertex_0: [vertices[0][0], vertices[0][1], vertices[0][2]],
            vertex_1: [vertices[1][0], vertices[1][1], vertices[1][2]],
            vertex_2: [vertices[2][0], vertices[2][1], vertices[2][2]],

            material,
            padding_0: 0,
            padding_1: 0,
        }
    }

    pub fn into_vertices(&self) -> [Vertex; 3] {
        let e0 = [
            self[1][0] - self[0][0], //
            self[1][1] - self[0][1], //
            self[1][2] - self[0][2], //
        ];

        let e1 = [
            self[2][0] - self[0][0], //
            self[2][1] - self[0][1], //
            self[2][2] - self[0][2], //
        ];

        let normal = [
            (e0[1] * e1[2]) - (e0[2] * e1[1]), //
            (e0[2] * e1[0]) - (e0[0] * e1[2]), //
            (e0[0] * e1[1]) - (e0[1] * e1[0]), //
        ];

        let normal_len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        let normal = [normal[0] / normal_len, normal[1] / normal_len, normal[2] / normal_len];

        [
            Vertex {
                position: self[0],
                normal,
                material: self.material,
            },
            Vertex {
                position: self[1],
                normal,
                material: self.material,
            },
            Vertex {
                position: self[2],
                normal,
                material: self.material,
            },
        ]
    }
}

pub fn triangles_into_vertices<const N: usize>(triangles: &[Triangle; N]) -> [Vertex; N * 3] {
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
