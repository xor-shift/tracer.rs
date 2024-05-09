mod extent;
mod path;
mod threaded;

use cgmath::InnerSpace;
pub use path::Path;
pub use threaded::ThreadedOctreeNode;

use self::{extent::DiscreteExtent, path::Turn};

#[derive(Clone, Copy, PartialEq, Eq)]
enum NodeState<T: Sized> {
    Sentinel,
    Node { children_start: usize },
    Leaf { material: T },
}

struct Node<T: Sized> {
    state: NodeState<T>,

    path: Path<24>,
    parent: usize,

    orphan: bool,
}

impl<T: Sized> Node<T> {
    pub fn make_child(turn: Turn) -> Node<T> {
        todo!();
    }
}

impl<T: Sized> std::default::Default for Node<T> {
    fn default() -> Self {
        Self {
            state: NodeState::Sentinel,

            path: Path::new(),
            parent: 0,

            orphan: false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct StaticNodeData {
    path: Path<24>,
    parent: usize,
}

pub struct VoxelTree<T: Sized = [u32; 2]> {
    dimensions: [u32; 3],
    data: Vec<(NodeState<T>, StaticNodeData)>,
    max_depth: usize,
}

impl<T: Sized + Clone + Copy + PartialEq> VoxelTree<T> {}

// for clarity
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct NodeIndex(pub usize);

enum DFSOrder {
    PreOrder,
    InOrder,
    PostOrder,
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct ChildSelectionOrder(pub [usize; 8]);

impl From<[usize; 8]> for ChildSelectionOrder {
    fn from(value: [usize; 8]) -> Self { Self(value) }
}

impl ChildSelectionOrder {
    // 0
    pub const ORDER_LOOKING_AT_X: ChildSelectionOrder = ChildSelectionOrder([0, 2, 4, 6, 1, 3, 5, 7]);
    pub const ORDER_LOOKING_AT_Y: ChildSelectionOrder = ChildSelectionOrder([0, 1, 4, 5, 2, 3, 6, 7]);
    pub const ORDER_LOOKING_AT_Z: ChildSelectionOrder = ChildSelectionOrder([0, 1, 2, 3, 4, 5, 6, 7]);
    pub const ORDER_LOOKING_AT_2_LAST: ChildSelectionOrder = ChildSelectionOrder([5, 1, 4, 7, 0, 3, 6, 2]);
    pub const ORDER_LOOKING_AT_3_LAST: ChildSelectionOrder = ChildSelectionOrder([4, 0, 5, 6, 1, 2, 7, 3]);
    pub const ORDER_LOOKING_AT_6_LAST: ChildSelectionOrder = ChildSelectionOrder([1, 0, 3, 5, 2, 4, 7, 6]);
    pub const ORDER_LOOKING_AT_7_LAST: ChildSelectionOrder = ChildSelectionOrder([0, 1, 2, 5, 3, 4, 6, 7]);

    // 7
    pub const ORDER_LOOKING_AT_NEG_X: ChildSelectionOrder = Self::ORDER_LOOKING_AT_X.reversed();
    pub const ORDER_LOOKING_AT_NEG_Y: ChildSelectionOrder = Self::ORDER_LOOKING_AT_Y.reversed();
    pub const ORDER_LOOKING_AT_NEG_Z: ChildSelectionOrder = Self::ORDER_LOOKING_AT_Z.reversed();
    pub const ORDER_LOOKING_AT_5_LAST: ChildSelectionOrder = Self::ORDER_LOOKING_AT_2_LAST.reversed();
    pub const ORDER_LOOKING_AT_4_LAST: ChildSelectionOrder = Self::ORDER_LOOKING_AT_3_LAST.reversed();
    pub const ORDER_LOOKING_AT_1_LAST: ChildSelectionOrder = Self::ORDER_LOOKING_AT_7_LAST.reversed();
    pub const ORDER_LOOKING_AT_0_LAST: ChildSelectionOrder = Self::ORDER_LOOKING_AT_6_LAST.reversed();

    pub const fn reversed(self) -> ChildSelectionOrder {
        let mut ret = ChildSelectionOrder([0; 8]);

        let mut i = 0;
        loop {
            ret.0[i] = self.0[7 - i];
            i += 1;
        }

        ret
    }

    /// This function is here to serve as a reference
    pub fn decide_order(v: cgmath::Vector3<f32>) -> usize {
        let characteristic_vectors = [
            cgmath::vec3(1., 0., 0.),
            cgmath::vec3(0., 1., 0.),
            cgmath::vec3(0., 0., 1.),
            cgmath::vec3(-1., 1., -1.).normalize(),
            cgmath::vec3(1., 1., -1.).normalize(),
            cgmath::vec3(-1., 1., 1.).normalize(),
            cgmath::vec3(1., 1., 1.).normalize(),
        ];

        let scores = characteristic_vectors.map(|cv| v.dot(cv).abs());

        let max = scores //
            .into_iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        if max.1 < 0. {
            max.0 + 7
        } else {
            max.0
        }
    }

    /// This function is here to serve as a reference much like decide_order,
    /// however, this one's goal is to be fast, and not to be accurate.
    #[allow(confusable_idents)]
    #[allow(mixed_script_confusables)]
    pub fn decide_fast(v: cgmath::Vector3<f32>) -> usize {
        let determine_largest = |x, y, z| {
            // slow version (and the logic behind this):
            /*match (x > y, x > z, z > y) {
                (false, false, false) => 1,             // y >= x, z >= x, y => z: y >= z >= x
                (false, false, true) => 2,              // y >= x, z >= x, z >  y: z >  y >= x
                (false, true, false) => 1,              // y >= x, x >  z, y => z: y >= x >  z
                (false, true, true) => unreachable!(),  // y >= x, x >  z, z >  y: impossible
                (true, false, false) => unreachable!(), // x >  y, z >= x, y => z: impossible
                (true, false, true) => 2,               // x >  y, z >= x, z >  y: z >= x > y
                (true, true, false) => 0,               // x >  y, x >  z, y => z: x > y >= z
                (true, true, true) => 0,                // x >  y, x >  z, z >  y: x > z > y
            };*/

            let (a, b, c) = (x > y, x > z, z > y);
            return (!a && !c) as usize * 1 + (!b && c) as usize * 2;
        };

        let abs_v: cgmath::Vector3<f32> = v.map(|v| v.abs());
        let largest_axis = determine_largest(abs_v.x, abs_v.y, abs_v.z);
        let corrected_largest_axis = if v[largest_axis] < 0. { largest_axis + 7 } else { largest_axis };

        // TODO: find a better metric
        //let is_corner = abs_v.dot(cgmath::vec3(1., 1., 1.).normalize()) > 0.87;
        let is_corner = (abs_v.x + abs_v.y + abs_v.z) > 1.5;

        /*let corner_if_corner = match (v.x > 0., v.y > 0., v.z > 0.) {
            (false, false, false) => 13, // 0 last
            (true, false, false) => 12,  // 1 last
            (false, true, false) => 3,   // 2 last
            (true, true, false) => 4,    // 3 last
            (false, false, true) => 11,
            (true, false, true) => 10,
            (false, true, true) => 5,
            (true, true, true) => 6,
        };*/

        let corner_if_corner = ((v.y <= 0.) as i32 * 10 - ((v.x > 0.) as i32 + (v.z > 0.) as i32 * 2)).abs() as usize + 3;

        if is_corner {
            corner_if_corner
        } else {
            corrected_largest_axis
        }
    }
}

impl std::ops::Index<usize> for ChildSelectionOrder {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output { &self.0[index] }
}

impl std::ops::IndexMut<usize> for ChildSelectionOrder {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output { &mut self.0[index] }
}

impl<T: Sized + Clone + Copy> VoxelTree<T> {
    pub fn new(dimensions: [u32; 3]) -> VoxelTree<T> {
        Self {
            dimensions,
            data: vec![(NodeState::Sentinel, StaticNodeData { path: Path::new(), parent: 0 })],
            max_depth: 0,
        }
    }

    fn reverse_bfs<Fun: FnMut(&VoxelTree<T>, usize)>() {}

    fn bfs<Fun: FnMut(&VoxelTree<T>, usize)>(&self, mut fun: Fun) {
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(0);

        while let Some(cur_index) = queue.pop_front() {
            fun(self, cur_index);

            match self.data[cur_index].0 {
                NodeState::Node { children_start } => {
                    for i in 0..=7 {
                        if matches!(self.data[children_start + i].0, NodeState::Sentinel {}) {
                            continue;
                        }

                        queue.push_back(children_start + i);
                    }
                }
                _ => {}
            }
        }
    }

    fn dfs_with_order<Fun: FnMut(&VoxelTree<T>, usize)>(&self, order: ChildSelectionOrder, mut fun: Fun) {
        let mut stack = vec![0];

        while let Some(cur_index) = stack.pop() {
            fun(self, cur_index);

            match self.data[cur_index].0 {
                NodeState::Node { children_start } => {
                    for i in order.0.into_iter().rev() {
                        if matches!(self.data[children_start + i].0, NodeState::Sentinel {}) {
                            continue;
                        }

                        stack.push(children_start + i);
                    }
                }
                _ => {}
            }
        }
    }

    fn dfs<Fun: FnMut(&VoxelTree<T>, usize)>(&self, mut fun: Fun) { self.dfs_with_order([0, 1, 2, 3, 4, 5, 6, 7].into(), fun) }

    fn parent_of(&self, index: usize) -> Option<usize> {
        let (_node, aux_data) = self.data[index];
        if aux_data.parent == index {
            None
        } else {
            Some(aux_data.parent)
        }
    }

    /// returns the next sibling index that comes after the node specified by `after`
    fn next_sibling_under_with_order(&self, order: ChildSelectionOrder, index: usize, after: usize) -> Option<usize> {
        let cs = if let NodeState::Node { children_start } = self.data[index].0 {
            children_start
        } else {
            return None;
        };

        let mut encountered = false;
        for i in 0..8 {
            let candidate = cs + order[i];

            if candidate == after {
                encountered = true;
                continue;
            }

            if !encountered {
                continue;
            }

            if matches!(self.data[candidate].0, NodeState::Sentinel) {
                continue;
            }

            return Some(candidate);
        }

        None
    }

    fn next_sibling_under(&self, index: usize, after: usize) -> Option<usize> { self.next_sibling_under_with_order([0, 1, 2, 3, 4, 5, 6, 7].into(), index, after) }

    fn first_child_of_with_order(&self, order: ChildSelectionOrder, index: usize) -> usize {
        let cs = if let NodeState::Node { children_start } = self.data[index].0 {
            children_start
        } else {
            panic!("first_child_of called on a leaf or a sentinel");
        };

        for i in order.0 {
            let candidate = cs + i;

            if matches!(self.data[candidate].0, NodeState::Sentinel) {
                continue;
            }

            return candidate;
        }

        panic!("invariant violated: node has no children");
    }

    fn first_child_of(&self, index: usize) -> usize { self.first_child_of_with_order([0, 1, 2, 3, 4, 5, 6, 7].into(), index) }

    fn extend_node(&mut self, index: usize) -> usize {
        let children_start = self.data.len();

        self.max_depth = self.max_depth.max(self.data[index].1.path.depth());

        self.data[index] = (NodeState::Node { children_start }, self.data[index].1);
        self.data.extend_from_slice(
            &[(
                NodeState::Sentinel,
                StaticNodeData {
                    path: self.data[index].1.path,
                    parent: index,
                },
            ); 8],
        );

        for i in 0..8 {
            self.data[children_start + i].1.path.take_turn(i.into());
        }

        children_start
    }

    pub fn set_voxel_path(&mut self, at: Path, material: T) -> color_eyre::Result<()> {
        let mut next_node_index = 0;
        for turn in at {
            let children_start = match self.data[next_node_index].0 {
                NodeState::Sentinel => self.extend_node(next_node_index),
                NodeState::Node { children_start } => children_start,
                NodeState::Leaf { material: _ } => {
                    return Err(eyre::eyre!("cannot insert a voxel as the child of another"));
                }
            };

            self.data[next_node_index].0 = NodeState::Node { children_start };

            let child_at_turn = children_start + turn as u8 as usize;
            next_node_index = child_at_turn;
        }

        if !matches!(self.data[next_node_index].0, NodeState::Sentinel {}) {
            return Err(eyre::eyre!("cannot overwrite a voxel or a node"));
        }

        self.data[next_node_index].0 = NodeState::Leaf { material };

        Ok(())
    }

    pub fn set_voxel(&mut self, at: cgmath::Point3<u32>, material: T) -> color_eyre::Result<()> {
        let path = DiscreteExtent {
            min: (0, 0, 0).into(),
            max: cgmath::Point3::from(self.dimensions).cast().unwrap() - cgmath::vec3(1, 1, 1),
        }
        .figure_out_path(at.cast().unwrap());

        self.set_voxel_path(path, material)
    }

    pub fn len(&self) -> usize { self.data.len() }

    pub fn depth(&self) -> usize { self.max_depth }
}

impl VoxelTree<[u32; 2]> {
    /// in general:
    /// - a node's miss link is towards the next sibling OR to the parent's miss link OR to the sentinel node
    /// - a node's hit link is towards the first child OR to the parent's miss link
    ///
    /// if the threaded graph is stored in the tree's DFS order (which is the representation used):
    /// - if the node is a leaf, the miss link is towards the next node (which goes towards the sentinel if the leaf represents the last node)
    /// - if the node is a branch, the miss link is towards the next sibling OR to the parent's miss link OR to the sentinel node <-- only this case matters
    /// - if the node is a leaf, the hit link is identical to the miss link
    /// - if the node is a branch, the hit link is towards the next node
    ///
    /// while the branch elements in the returned vector contain offsets, if
    /// the offset is identical to u32::MAX, it ought not be considered an
    /// offset but as a reference to the sentinel node.
    ///
    /// the time complexity of this function is O(n) and the memory complexity
    /// is O(n) if one considers the return value, O(logn) otherwise (for the
    /// traversals); use it sparingly (might implement things that operate
    /// directly on the threadedrepresentation but don't hold your breath on
    /// that)
    ///
    /// wow what a wall of text, me.. will you please stop yapping
    pub fn thread_with_order(&self, order: ChildSelectionOrder) -> Vec<ThreadedOctreeNode> {
        const SENTINEL_OFFSET: u32 = u32::MAX;

        let mut ret_graph = vec![];
        let mut graph_to_tree_map = vec![];

        self.dfs_with_order(order, |this: &VoxelTree, tree_index| {
            let (cur_node, cur_aux) = this.data[tree_index];

            let graph_node_to_push = match cur_node {
                NodeState::Sentinel => {
                    return;
                }
                NodeState::Leaf { material } => ThreadedOctreeNode::new_leaf(cur_aux.path.truncating_cap_cast(), material),
                NodeState::Node { .. } => ThreadedOctreeNode::new_node(cur_aux.path.truncating_cap_cast(), SENTINEL_OFFSET),
            };

            graph_to_tree_map.push(tree_index);
            ret_graph.push(graph_node_to_push);
        });

        let mut tree_to_graph_map = vec![0; self.data.len()];
        for (graph_idx, &tree_idx) in graph_to_tree_map.iter().enumerate() {
            tree_to_graph_map[tree_idx] = graph_idx;
        }
        let tree_to_graph_map = tree_to_graph_map;

        self.dfs_with_order(order, |this, tree_index| {
            // skip root
            let tree_parent_index = if let Some(v) = this.parent_of(tree_index) {
                v
            } else {
                return;
            };

            let (cur_tree_node, cur_aux) = this.data[tree_index];

            if let NodeState::Node { .. } = cur_tree_node {
                let graph_parent = ret_graph[tree_to_graph_map[tree_parent_index]];
                let graph_parent_miss_index = graph_parent.miss_link(tree_to_graph_map[tree_parent_index] as u32);
                let miss_offset_if_last_sibling = if graph_parent_miss_index == SENTINEL_OFFSET {
                    SENTINEL_OFFSET
                } else {
                    graph_parent_miss_index - tree_to_graph_map[tree_index] as u32
                };

                let miss_offset = this
                    .next_sibling_under_with_order(order, tree_parent_index, tree_index)
                    .map(|sibling_index| {
                        let sibling_graph_index = tree_to_graph_map[sibling_index];
                        (sibling_graph_index - tree_to_graph_map[tree_index]) as u32
                    })
                    .unwrap_or(miss_offset_if_last_sibling);

                ret_graph[tree_to_graph_map[tree_index]].set_miss_offset(miss_offset);
            }
        });

        ret_graph
    }

    pub fn thread(&self) -> Vec<ThreadedOctreeNode> { self.thread_with_order([0, 1, 2, 3, 4, 5, 6, 7].into()) }
}

#[rustfmt::skip]
const TEST_VOXMAP_DATA: [u8; 64] = [
    0b11000000, 0b11000000, 0b00011000, 0b00111100, 0b00111100, 0b00011000, 0b00000000, 0b00000000,
    0b11000000, 0b11011000, 0b00111100, 0b01111110, 0b01111110, 0b00111100, 0b00011000, 0b00000000,
    0b00011000, 0b00111100, 0b01111110, 0b11111111, 0b11111111, 0b01111110, 0b00111100, 0b00011000,
    0b00100100, 0b01100110, 0b11100111, 0b11100111, 0b11100111, 0b11100111, 0b01100110, 0b00100100,
    0b00100100, 0b01100110, 0b11100111, 0b11100111, 0b11100111, 0b11100111, 0b01100110, 0b00100100,
    0b00011000, 0b00111100, 0b01111110, 0b11111111, 0b11111111, 0b01111110, 0b00111100, 0b00011000,
    0b00000000, 0b00011000, 0b00111100, 0b01111110, 0b01111110, 0b00111100, 0b00011000, 0b00000000,
    0b00000000, 0b00000000, 0b00011000, 0b00111100, 0b00111100, 0b00011000, 0b00000000, 0b00000000,
];

fn get_test_voxmap() -> impl Iterator<Item = (i64, i64, i64)> {
    TEST_VOXMAP_DATA
        .into_iter()
        .enumerate()
        .map(|(idx, v)| {
            let z = idx / 8;
            let y = idx % 8;

            let mut ret = [None; 8];

            for x in 0..8 {
                if ((v >> x) & 1) == 0 {
                    continue;
                }

                ret[x] = Some((x as i64, y as i64, z as i64));
            }

            ret
        })
        .flatten()
        .filter_map(|v| v)
}

pub fn get_test_voxel_tree() -> VoxelTree<[u32; 2]> {
    let mut tree = VoxelTree::new([8; 3]);

    let gray = [0x01333333, 0x00000000];
    let lgray = [0x01555555, 0x00000000];
    let red = [0x01FF0000, 0x00000000];
    let green = [0x0100FF00, 0x00000000];
    let blue = [0x010000FF, 0x00000000];

    for coords in get_test_voxmap() {
        let material: [u32; 2] = match coords {
            (0..2, 0..2, 0..2) => gray, // gray
            (2..4, 0..2, 0..2) => red,  // red
            (0..2, 2..4, 0..2) => red,
            (2..4, 2..4, 0..2) => red,
            (0..2, 0..2, 2..4) => red,
            (2..4, 0..2, 2..4) => red,
            (0..2, 2..4, 2..4) => red,
            (2..4, 2..4, 2..4) => red,

            (4..8, 0..4, 0..4) => green,
            (0..4, 4..8, 0..4) => blue,
            (4..8, 4..8, 0..4) => red,
            (0..4, 0..4, 4..8) => green,
            (4..8, 0..4, 4..8) => blue,
            (0..4, 4..8, 4..8) => red,
            (4..8, 4..8, 4..8) => green,

            (8.., _, _) => unreachable!(),
            (_, 8.., _) => unreachable!(),
            (_, _, 8..) => unreachable!(),

            (..0, _, _) => unreachable!(),
            (_, ..0, _) => unreachable!(),
            (_, _, ..0) => unreachable!(),
        };

        assert!(tree
            .set_voxel_path(
                DiscreteExtent {
                    min: cgmath::point3(0, 0, 0),
                    max: cgmath::point3(7, 7, 7),
                }
                .figure_out_path(coords.into()),
                material
            )
            .is_ok());
    }

    tree
}

// print based testing lets goooo
#[cfg(test)]
mod test {
    use crate::basic_octree::extent::DiscreteExtent;

    use super::*;

    #[test]
    fn test_vt_0() {
        let mut tree = VoxelTree::new([8; 3]);

        assert!(tree.set_voxel_path(Path::from_digits(&[0]), [0, 0]).is_ok());
        assert!(tree.data.len() == 9);
        assert!(matches!(tree.data[0].0, NodeState::Node { children_start: 1 }));
        assert!(matches!(tree.data[1].0, NodeState::Leaf { material: [0, 0] }));

        assert!(tree.set_voxel_path(Path::from_digits(&[1]), [0, 1]).is_ok());
        assert!(tree.data.len() == 9);
        assert!(matches!(tree.data[2].0, NodeState::Leaf { material: [0, 1] }));

        println!("{:?}", tree);
    }

    #[test]
    fn test_vt_1() {
        let mut tree = VoxelTree::new([8; 3]);

        assert!(tree.set_voxel_path(Path::from_digits(&[0, 1, 2]), [0, 0]).is_ok());
        assert!(tree.data.len() == 25);

        println!("{:?}", tree);
    }

    #[test]
    fn test_ve_2() {
        let tree = get_test_voxel_tree();

        println!("{:?}", tree);
        println!("{:#?}", tree.thread());
    }

    #[test]
    fn test_ve_3() {
        let mut tree = VoxelTree::new([2; 3]);

        for i in 0..8 {
            assert!(tree.set_voxel((i % 2, (i / 2) % 2, i / 4).into(), [0; 2]).is_ok());
        }

        println!("{:?}", tree);
        println!("{:#?}", tree.thread());
    }

    #[test]
    fn test_order_decision() {
        assert_eq!(0, ChildSelectionOrder::decide_order(cgmath::vec3(1., 0., 0.).normalize()));
        assert_eq!(0, ChildSelectionOrder::decide_order(cgmath::vec3(1., 0., 0.1).normalize()));
        assert_eq!(0, ChildSelectionOrder::decide_order(cgmath::vec3(1., 0.1, 0.).normalize()));
        assert_eq!(0, ChildSelectionOrder::decide_order(cgmath::vec3(1., 0.1, 0.1).normalize()));
        assert_eq!(1, ChildSelectionOrder::decide_order(cgmath::vec3(-1., 0., 0.).normalize()));
        assert_eq!(1, ChildSelectionOrder::decide_order(cgmath::vec3(-1., 0., 0.1).normalize()));
        assert_eq!(1, ChildSelectionOrder::decide_order(cgmath::vec3(-1., 0.1, 0.).normalize()));
        assert_eq!(1, ChildSelectionOrder::decide_order(cgmath::vec3(-1., 0.1, 0.1).normalize()));
        assert_eq!(13, ChildSelectionOrder::decide_order(cgmath::vec3(1., 1., 1.).normalize()));
    }
}

impl std::fmt::Debug for VoxelTree<[u32; 2]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write;

        // ├╰│
        let mut stack = Vec::new();
        stack.push((0, 0, Path::<24>::new()));

        let mut branch_states = 0;

        while let Some((cur_index, remaining_siblings, path_upto)) = stack.pop() {
            let is_root = cur_index == 0;

            let node = self.data[cur_index].0;

            if !is_root {
                let branch_char = if remaining_siblings == 0 { '╰' } else { '├' };
                if remaining_siblings == 0 {
                    branch_states &= !(1 << path_upto.depth());
                }

                for chk_depth in 1..path_upto.depth() {
                    let depth_has_branches_remaining = (branch_states & (1 << chk_depth)) != 0;
                    let branch_char = if depth_has_branches_remaining { '│' } else { ' ' };
                    f.write_char(branch_char)?;
                }

                f.write_char(branch_char)?;
            }

            match node {
                NodeState::Sentinel => unreachable!(),
                NodeState::Leaf { material } => {
                    f.write_fmt(format_args!("🌿 @ {}: [{:08X}, {:08X}]\n", cur_index, material[0], material[1]))?;
                }
                NodeState::Node { children_start } => {
                    let symbol = if is_root { '🌲' } else { '🪵' };
                    f.write_fmt(format_args!("{symbol} @ {}: {}\n", cur_index, children_start,))?;

                    let child_count = (&self.data[children_start..children_start + 8]) //
                        .iter()
                        .filter(|v| !matches!(v.0, NodeState::Sentinel {}))
                        .count();

                    let mut remaining = child_count;

                    if remaining != 0 {
                        branch_states |= 1 << (path_upto.depth() + 1);
                    }

                    for i in 0..=7 {
                        let i = 7 - i;
                        //if !branch_data.is_child(i) {
                        if matches!(self.data[children_start + i].0, NodeState::Sentinel {}) {
                            continue;
                        }

                        remaining -= 1;
                        stack.push((children_start + i, child_count - remaining - 1, path_upto.after_taken_turn(path::Turn::from(i))));
                    }
                }
            }
        }

        Ok(())
    }
}
