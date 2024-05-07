mod extent;
mod path;
mod threaded;

use std::collections::HashMap;

pub use path::Path;
pub use threaded::ThreadedOctreeNode;

use self::extent::DiscreteExtent;

#[derive(Clone, Copy, PartialEq, Eq)]
enum TreeNode<T: Sized> {
    Sentinel,
    Node { children_start: usize },
    Leaf { material: T },
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct AuxilliaryTreeData {
    path: Path<24>,
    parent: usize,
}

pub struct VoxelTree<T: Sized = [u32; 2]> {
    dimensions: [u32; 3],
    data: Vec<(TreeNode<T>, AuxilliaryTreeData)>,
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

impl<T: Sized + Clone + Copy> VoxelTree<T> {
    pub fn new(dimensions: [u32; 3]) -> VoxelTree<T> {
        Self {
            dimensions,
            data: vec![(TreeNode::Sentinel, AuxilliaryTreeData { path: Path::new(), parent: 0 })],
        }
    }

    fn bfs<Fun: FnMut(&VoxelTree<T>, usize)>(&self, mut fun: Fun) {
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(0);

        while let Some(cur_index) = queue.pop_front() {
            fun(self, cur_index);

            match self.data[cur_index].0 {
                TreeNode::Node { children_start } => {
                    for i in 0..=7 {
                        if matches!(self.data[children_start + i].0, TreeNode::Sentinel {}) {
                            continue;
                        }

                        queue.push_back(children_start + i);
                    }
                }
                _ => {}
            }
        }
    }

    fn dfs<Fun: FnMut(&VoxelTree<T>, usize)>(&self, mut fun: Fun) {
        let mut stack = vec![0];

        while let Some(cur_index) = stack.pop() {
            fun(self, cur_index);

            match self.data[cur_index].0 {
                TreeNode::Node { children_start } => {
                    for i in 0..=7 {
                        if matches!(self.data[children_start + 7 - i].0, TreeNode::Sentinel {}) {
                            continue;
                        }

                        stack.push(children_start + 7 - i);
                    }
                }
                _ => {}
            }
        }
    }

    fn parent_of(&self, index: usize) -> Option<usize> {
        let (_node, aux_data) = self.data[index];
        if aux_data.parent == index {
            None
        } else {
            Some(aux_data.parent)
        }
    }

    /// returns the next sibling index that comes after the node specified by `after`
    fn next_sibling_under(&self, index: usize, after: usize) -> Option<usize> {
        let cs = if let TreeNode::Node { children_start } = self.data[index].0 {
            children_start
        } else {
            return None;
        };

        for candidate in (after + 1)..cs + 8 {
            if matches!(self.data[candidate].0, TreeNode::Sentinel) {
                continue;
            }

            return Some(candidate);
        }

        None
    }

    fn first_child_of(&self, index: usize) -> usize {
        let cs = if let TreeNode::Node { children_start } = self.data[index].0 {
            children_start
        } else {
            panic!("first_child_of called on a leaf or a sentinel");
        };

        for i in 0..8 {
            let candidate = cs + i;

            if matches!(self.data[candidate].0, TreeNode::Sentinel) {
                continue;
            }

            return candidate;
        }

        panic!("invariant violated: node has no children");
    }

    fn extend_node(&mut self, index: usize) -> usize {
        let children_start = self.data.len();

        self.data[index] = (TreeNode::Node { children_start }, self.data[index].1);
        self.data.extend_from_slice(
            &[(
                TreeNode::Sentinel,
                AuxilliaryTreeData {
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
                TreeNode::Sentinel => self.extend_node(next_node_index),
                TreeNode::Node { children_start } => children_start,
                TreeNode::Leaf { material: _ } => {
                    return Err(eyre::eyre!("cannot insert a voxel as the child of another"));
                }
            };

            self.data[next_node_index].0 = TreeNode::Node { children_start };

            let child_at_turn = children_start + turn as u8 as usize;
            next_node_index = child_at_turn;
        }

        if !matches!(self.data[next_node_index].0, TreeNode::Sentinel {}) {
            return Err(eyre::eyre!("cannot overwrite a voxel or a node"));
        }

        self.data[next_node_index].0 = TreeNode::Leaf { material };

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
    pub fn thread(&self) -> Vec<ThreadedOctreeNode> {
        const SENTINEL_OFFSET: u32 = u32::MAX;

        let mut ret_graph = vec![];
        let mut graph_to_tree_map = vec![];

        self.dfs(|this: &VoxelTree, tree_index| {
            let (cur_node, cur_aux) = this.data[tree_index];

            let graph_node_to_push = match cur_node {
                TreeNode::Sentinel => {
                    return;
                }
                TreeNode::Leaf { material } => ThreadedOctreeNode::new_leaf(cur_aux.path.truncating_cap_cast(), material),
                TreeNode::Node { .. } => ThreadedOctreeNode::new_node(cur_aux.path.truncating_cap_cast(), SENTINEL_OFFSET),
            };

            graph_to_tree_map.push(tree_index);
            ret_graph.push(graph_node_to_push);
        });

        let mut tree_to_graph_map = vec![0; self.data.len()];
        for (graph_idx, &tree_idx) in graph_to_tree_map.iter().enumerate() {
            tree_to_graph_map[tree_idx] = graph_idx;
        }
        let tree_to_graph_map = tree_to_graph_map;

        self.dfs(|this, tree_index| {
            // skip root
            let tree_parent_index = if let Some(v) = this.parent_of(tree_index) {
                v
            } else {
                return;
            };

            let (cur_tree_node, cur_aux) = this.data[tree_index];

            if let TreeNode::Node { .. } = cur_tree_node {
                let graph_parent = ret_graph[tree_to_graph_map[tree_parent_index]];
                let graph_parent_miss_index = graph_parent.miss_link(tree_to_graph_map[tree_parent_index] as u32);
                let miss_offset_if_last_sibling = if graph_parent_miss_index == SENTINEL_OFFSET {
                    SENTINEL_OFFSET
                } else {
                    graph_parent_miss_index - tree_to_graph_map[tree_index] as u32
                };

                let miss_offset = this
                    .next_sibling_under(tree_parent_index, tree_index)
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
        assert!(matches!(tree.data[0].0, TreeNode::Node { children_start: 1 }));
        assert!(matches!(tree.data[1].0, TreeNode::Leaf { material: [0, 0] }));

        assert!(tree.set_voxel_path(Path::from_digits(&[1]), [0, 1]).is_ok());
        assert!(tree.data.len() == 9);
        assert!(matches!(tree.data[2].0, TreeNode::Leaf { material: [0, 1] }));

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
                TreeNode::Sentinel => unreachable!(),
                TreeNode::Leaf { material } => {
                    f.write_fmt(format_args!("🌿 @ {}: [{:08X}, {:08X}]\n", cur_index, material[0], material[1]))?;
                }
                TreeNode::Node { children_start } => {
                    let symbol = if is_root { '🌲' } else { '🪵' };
                    f.write_fmt(format_args!("{symbol} @ {}: {}\n", cur_index, children_start,))?;

                    let child_count = (&self.data[children_start..children_start + 8]) //
                        .iter()
                        .filter(|v| !matches!(v.0, TreeNode::Sentinel {}))
                        .count();

                    let mut remaining = child_count;

                    if remaining != 0 {
                        branch_states |= 1 << (path_upto.depth() + 1);
                    }

                    for i in 0..=7 {
                        let i = 7 - i;
                        //if !branch_data.is_child(i) {
                        if matches!(self.data[children_start + i].0, TreeNode::Sentinel {}) {
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
