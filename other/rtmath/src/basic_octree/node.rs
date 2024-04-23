use super::{extent::VoxelExtent, node_path::NodePath};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct TreeNodeIndex(pub usize);

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct TreeDataIndex(pub usize);

#[derive(Clone, Copy)]
pub(super) struct BranchData {
    pub children_start: TreeNodeIndex,
    pub children_mask: u8,
}

impl BranchData {
    pub fn child_count(&self) -> usize { self.children_mask.count_ones() as usize }

    pub fn is_child(&self, index: usize) -> bool { (self.children_mask & (1 << index)) != 0 }

    // leading 76543210 trailing
    //         01001101
    //         10110010

    pub fn leftmost_child_index(&self) -> usize { self.children_mask.trailing_zeros() as usize }

    pub fn is_leftmost_child(&self, index: usize) -> bool { self.leftmost_child_index() == index }

    pub fn is_rightmost_child(&self, index: usize) -> bool { (7 - self.children_mask.leading_zeros()) as usize == index }
}

#[derive(Clone, Copy)]
pub(super) struct LeafData {
    pub data_start: TreeDataIndex,
    pub data_length: usize,
}

#[derive(Clone, Copy)]
pub(super) enum NodeState {
    Branch(BranchData),
    Leaf(LeafData),
}

#[derive(Clone, Copy)]
pub(super) struct OctreeNode {
    pub path: NodePath,
    pub parent: Option<TreeNodeIndex>,

    pub extent: VoxelExtent,
    pub state: NodeState,
}

impl OctreeNode {
    pub fn new_root(data_length: usize, extent: VoxelExtent) -> OctreeNode {
        Self {
            path: NodePath::new(),
            parent: None,

            extent,
            state: NodeState::Leaf(LeafData {
                data_start: TreeDataIndex(0),
                data_length: data_length,
            }),
        }
    }

    pub fn is_leaf(&self) -> bool { matches!(self.state, NodeState::Leaf(_)) }
}

/// gpu-friendly graph/tree node with hit & miss links
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
#[repr(C)]
pub struct ThreadedOctreeNode {
    pub(super) path_pack: [u32; 2],

    pub(super) link_hit: u32,
    pub(super) link_miss: u32,

    pub(super) data_start: u32,
    pub(super) data_length: u32,
}

impl ThreadedOctreeNode {
    pub(super) fn imbue_path(mut self, path: NodePath) -> ThreadedOctreeNode {
        self.path_pack = path.pack();
        self
    }

    pub(super) fn get_path(&self) -> NodePath { NodePath::from_pack(self.path_pack) }
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
#[repr(C)]
pub struct NewThreadedOctreeNode {
    pub(super) path_pack: [u32; 2],

    pub(super) link_hit: u32,
    pub(super) link_miss: u32,

    pub(super) material_pack: [u32; 2], // only relevant if hit and miss links are the same
}

impl NewThreadedOctreeNode {
    pub const fn new_quick(path: &[u8], links: [u32; 2], material: [u32; 2]) -> NewThreadedOctreeNode {
        let mut ret_path = NodePath::<8>::new();

        let mut i = 0; // ugly loop because of const fn
        loop {
            if i == path.len() {
                break;
            }

            ret_path.take_turn(unsafe { std::mem::transmute(path[i]) });

            i += 1;
        }

        NewThreadedOctreeNode {
            path_pack: ret_path.pack(),
            link_hit: links[0],
            link_miss: links[1],
            material_pack: material,
        }
    }
}
