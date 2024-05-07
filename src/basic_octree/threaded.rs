use super::path::Path;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ThreadedOctreeNode {
    pub(super) path_pack: [u32; 2],
    pub(super) pack_links_material: [u32; 2],
}

impl ThreadedOctreeNode {
    pub fn is_leaf(&self) -> bool { (self.pack_links_material[0] >> 31) == 0 }

    pub fn material(&self) -> [u32; 2] { return [self.pack_links_material[0] & 0x7FFFFFFF, self.pack_links_material[1]] }

    pub fn hit_link(&self, self_index: u32) -> u32 { self_index + 1 }

    pub fn miss_link(&self, self_index: u32) -> u32 {
        if self.is_leaf() {
            self_index + 1
        } else if self.miss_offset() == u32::MAX {
            u32::MAX
        } else {
            self.miss_offset() + self_index
        }
    }

    pub fn path(&self) -> Path<16> { Path::from_data(&self.path_pack) }

    pub fn miss_offset(&self) -> u32 { self.pack_links_material[1] }

    pub fn set_miss_offset(&mut self, offset: u32) { self.pack_links_material[1] = offset; }
}

impl std::fmt::Debug for ThreadedOctreeNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_leaf() {
            f.write_fmt(format_args!("🌿 {:?} [{:08X}, {:08X}]", self.path(), self.material()[0], self.material()[1]))
        } else {
            f.write_fmt(format_args!("🪵 {:?} +{}", self.path(), self.pack_links_material[1]))
        }
    }
}

impl ThreadedOctreeNode {
    const fn pack_quick_path(path: &[u8]) -> [u32; 2] {
        let mut ret_path = Path::<16>::new();

        let mut i = 0; // ugly loop because of const fn
        loop {
            if i == path.len() {
                break;
            }

            ret_path.take_turn(unsafe { std::mem::transmute(path[i]) });

            i += 1;
        }

        *ret_path.data()
    }

    pub const fn new_quick_leaf(path: &[u8], material: [u32; 2]) -> ThreadedOctreeNode {
        ThreadedOctreeNode {
            path_pack: Self::pack_quick_path(path),
            pack_links_material: material,
        }
    }

    pub const fn new_quick_node(path: &[u8], offset: u32) -> ThreadedOctreeNode {
        ThreadedOctreeNode {
            path_pack: Self::pack_quick_path(path),
            pack_links_material: [0x80000000, offset],
        }
    }

    pub const fn new_leaf(path: Path<16>, material: [u32; 2]) -> ThreadedOctreeNode {
        ThreadedOctreeNode {
            path_pack: *path.data(),
            pack_links_material: material,
        }
    }

    pub const fn new_node(path: Path<16>, offset: u32) -> ThreadedOctreeNode {
        ThreadedOctreeNode {
            path_pack: *path.data(),
            pack_links_material: [0x80000000, offset],
        }
    }
}
