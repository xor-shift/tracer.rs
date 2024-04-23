mod extent;
mod node;
mod node_path;

use std::{collections::HashMap, fmt::Write};

use extent::VoxelExtent;
use node_path::NodePath;

use node::{BranchData, LeafData, NodeState, OctreeNode, TreeDataIndex, TreeNodeIndex};
pub use node::{NewThreadedOctreeNode, ThreadedOctreeNode};

pub struct Octree<T: Sized> {
    data: Vec<(cgmath::Point3<i64>, T)>,
    nodes: Vec<OctreeNode>,
}

impl<T: Sized> Octree<T> {
    pub fn new(data: Vec<(cgmath::Point3<i64>, T)>) -> Octree<T> {
        const PT_MIN: cgmath::Point3<i64> = cgmath::point3(i64::MIN, i64::MIN, i64::MIN);
        const PT_MAX: cgmath::Point3<i64> = cgmath::point3(i64::MAX, i64::MAX, i64::MAX);

        let extent = data.iter().map(|v| v.0).fold((PT_MAX, PT_MIN), |acc, v| (acc.0.zip(v, i64::min), acc.1.zip(v, i64::max)));
        let extent = extent.into();

        Self {
            nodes: vec![OctreeNode::new_root(data.len(), extent)],
            data,
        }
    }

    fn node(&self, index: TreeNodeIndex) -> &OctreeNode { &self.nodes[index.0] }
    fn node_mut(&mut self, index: TreeNodeIndex) -> &mut OctreeNode { &mut self.nodes[index.0] }

    fn split_node(&mut self, index: TreeNodeIndex) -> Option<BranchData> {
        let leaf_data = match self.node(index).state {
            NodeState::Branch(_) => return None,
            NodeState::Leaf(v) => v,
        };

        if leaf_data.data_length < 12 {
            return None;
        }

        let partitions = self.node(index).extent.clone().partition(&mut self.data[leaf_data.data_start.0..leaf_data.data_start.0 + leaf_data.data_length]);

        let mut new_nodes = [std::mem::MaybeUninit::uninit(); 8];

        let mut children_mask = 0;

        for i in 0..=7 {
            let data_start = match i {
                0 => 0,
                i => partitions[i - 1],
            };
            let data_start = TreeDataIndex(data_start);

            let data_length = match i {
                0 => partitions[0],
                1..=6 => partitions[i] - partitions[i - 1],
                7 => leaf_data.data_length - partitions[6],
                8.. => unreachable!(),
            };

            let v = OctreeNode {
                path: self.node(index).path.after_taken_turn(unsafe { std::mem::transmute(i as u8) }),
                parent: Some(index),

                extent: self.node(index).extent.get_split(i),
                state: NodeState::Leaf(LeafData { data_start, data_length }),
            };

            children_mask >>= 1;
            if data_length != 0 {
                children_mask |= 0x80;
            }

            new_nodes[i] = std::mem::MaybeUninit::new(v);
        }

        // cba
        let new_nodes = new_nodes.map(|v| unsafe { v.assume_init() });

        let new_state = BranchData {
            children_start: TreeNodeIndex(self.nodes.len()),
            children_mask,
        };

        self.node_mut(index).state = NodeState::Branch(new_state);

        self.nodes.extend_from_slice(&new_nodes);

        Some(new_state)
    }

    pub fn split_until(&mut self, until_depth: usize) {
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(TreeNodeIndex(0));

        let push_children = |this: &mut Self, queue: &mut std::collections::VecDeque<_>, branch_data: BranchData, current_index| {
            if this.node(current_index).path.depth() >= until_depth {
                return;
            }

            for i in 0..=7 {
                if !branch_data.is_child(i) {
                    continue;
                }

                queue.push_back(TreeNodeIndex(branch_data.children_start.0 + i));
            }
        };

        while let Some(current_index) = queue.pop_front() {
            match self.node(current_index).state {
                NodeState::Branch(branch_data) => {
                    push_children(self, &mut queue, branch_data, current_index);
                }
                NodeState::Leaf(_leaf_data) => {
                    let branch_data = if let Some(branch_data) = self.split_node(current_index) {
                        branch_data
                    } else {
                        continue;
                    };

                    push_children(self, &mut queue, branch_data, branch_data.children_start);
                }
            }
        }
    }

    fn bfs<Fun: FnMut(&Octree<T>, TreeNodeIndex)>(&self, mut fun: Fun) {
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(TreeNodeIndex(0));

        while let Some(cur_index) = queue.pop_front() {
            fun(self, cur_index);

            match self.node(cur_index).state {
                NodeState::Branch(branch_data) => {
                    for i in 0..=7 {
                        if !branch_data.is_child(i) {
                            continue;
                        }

                        queue.push_back(TreeNodeIndex(branch_data.children_start.0 + i));
                    }
                }
                _ => {}
            }
        }
    }

    //TODO: bad!
    fn build_index_lookup(&self) -> (HashMap<TreeNodeIndex, usize>, HashMap<usize, TreeNodeIndex>) {
        let mut indices = Vec::new();
        let mut last_index = 0;

        self.bfs(|this, cur_index| {
            indices.push(cur_index);
        });

        let mut forward_map = HashMap::new();
        let mut backward_map = HashMap::new();

        for (graph_index, tree_index) in indices.into_iter().enumerate() {
            forward_map.insert(tree_index, graph_index);
            backward_map.insert(graph_index, tree_index);
        }

        (forward_map, backward_map)
    }

    pub fn thread(&self) -> Vec<ThreadedOctreeNode> {
        let (fwd_lookup, bwd_lookup) = self.build_index_lookup();

        const SENTINEL_INDEX: u32 = u32::MAX;

        let mut ret = Vec::new();

        self.bfs(|this, cur_index| {
            let node = this.node(cur_index);
            let tree_parent = node.parent.map(|p| self.node(p));
            let graph_parent = node.parent.map(|v| *fwd_lookup.get(&v).unwrap()).map(|v| ret[v]);

            let have_sibling = tree_parent
                .map(|v| {
                    if let NodeState::Branch(branch_data) = v.state {
                        !branch_data.is_rightmost_child(node.path.last_turn().unwrap() as usize)
                    } else {
                        unreachable!()
                    }
                })
                .unwrap_or(false);

            // miss: sibling OR parent's miss OR sentinel
            // hit: leftmost child OR miss

            let parent_miss_link = graph_parent.map(|v: ThreadedOctreeNode| v.link_miss).unwrap_or(SENTINEL_INDEX);
            let link_miss = if have_sibling { ret.len() as u32 + 1 } else { parent_miss_link };

            let link_hit = match node.state {
                NodeState::Branch(branch_data) => {
                    let left_child_tnindex = TreeNodeIndex(branch_data.children_start.0 + branch_data.leftmost_child_index());
                    let left_child_graph_index = *fwd_lookup.get(&left_child_tnindex).unwrap();
                    left_child_graph_index as u32
                }
                NodeState::Leaf(leaf_data) => link_miss,
            };

            // println!("idx: {} ({})", cur_index.0, fwd_lookup.get(&cur_index).unwrap());

            let to_push = match node.state {
                NodeState::Branch(branch_data) => ThreadedOctreeNode {
                    path_pack: [0, 0],
                    link_hit,
                    link_miss,
                    data_start: 0,
                    data_length: 0,
                },
                NodeState::Leaf(leaf_data) => ThreadedOctreeNode {
                    path_pack: [0, 0],
                    link_hit,
                    link_miss,
                    data_start: leaf_data.data_start.0 as u32,
                    data_length: leaf_data.data_length as u32,
                },
            };

            ret.push(to_push.imbue_path(node.path));
        });

        return ret;
    }

    pub fn get_data(&self) -> &Vec<(cgmath::Point3<i64>, T)> { &self.data }
}

impl<T: Sized> std::fmt::Debug for Octree<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // ├╰│
        let mut stack = Vec::new();
        stack.push((TreeNodeIndex(0), 0));

        let mut branch_states = 1;

        f.write_str("🌲\n")?;

        while let Some((cur_index, remaining_siblings)) = stack.pop() {
            let node = self.node(cur_index);

            let branch_char = if remaining_siblings == 0 { '╰' } else { '├' };
            if remaining_siblings == 0 {
                branch_states &= !(1 << node.path.depth());
            }

            for chk_depth in 0..node.path.depth() {
                let depth_has_branches_remaining = (branch_states & (1 << chk_depth)) != 0;
                let branch_char = if depth_has_branches_remaining { '│' } else { ' ' };
                f.write_char(branch_char)?;
            }

            match node.state {
                NodeState::Leaf(leaf_data) => {
                    f.write_fmt(format_args!(
                        "{branch_char}🌿 @ {}: {}..{} ({})\n",
                        cur_index.0,
                        leaf_data.data_start.0,
                        leaf_data.data_start.0 + leaf_data.data_length,
                        leaf_data.data_length
                    ))?;
                }
                NodeState::Branch(branch_data) => {
                    f.write_fmt(format_args!(
                        "{branch_char}🪵 @ {}: {:08b} ({}) @ {}\n",
                        cur_index.0,
                        branch_data.children_mask,
                        branch_data.child_count(),
                        branch_data.children_start.0
                    ))?;

                    let mut remaining = branch_data.child_count();

                    if remaining != 0 {
                        branch_states |= 1 << (node.path.depth() + 1);
                    }

                    for i in 0..=7 {
                        let i = 7 - i;
                        if !branch_data.is_child(i) {
                            continue;
                        }

                        remaining -= 1;
                        stack.push((TreeNodeIndex(branch_data.children_start.0 + i), branch_data.child_count() - remaining - 1));
                    }
                }
            }
        }

        Ok(())
    }
}
