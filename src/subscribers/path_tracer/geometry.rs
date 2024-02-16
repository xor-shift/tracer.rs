#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GeometryElement {
    pack_0: [f32; 4],
    pack_1: [f32; 4],
    pack_2: [f32; 4],
    pack_3: [f32; 4],
}
