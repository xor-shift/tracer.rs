use super::{
    chunk::Chunk,
    data_iterator::{ChunkTree, RawChunk},
    transform::Transform,
};

pub struct State {
    origin: [i32; 3],
    transform: Transform,
}

impl std::default::Default for State {
    fn default() -> Self {
        Self {
            origin: [0, 0, 0],
            transform: Transform::identity(),
        }
    }
}

struct TraversalItem {
    depth: usize,
    index: usize,
    state: State,
}

pub struct IterationItem<'a> {
    pub depth: usize,
    pub state: State,
    pub raw_chunk: RawChunk<'a>,
    pub chunk: color_eyre::Result<Chunk<'a>>,
}

pub struct ChunkTreeIterator<'a, 'b> {
    tree: &'b ChunkTree<'a>,
    traversal_stack: Vec<TraversalItem>,

    errored: bool,
}

impl<'a, 'b> ChunkTreeIterator<'a, 'b> {
    pub fn new(tree: &'b ChunkTree<'a>) -> ChunkTreeIterator<'a, 'b> {
        let traversal_stack = tree
            .root_indices
            .iter()
            .rev()
            .map(|v| TraversalItem {
                depth: 0,
                index: *v,
                state: std::default::Default::default(),
            })
            .collect();

        Self { tree, traversal_stack, errored: false }
    }
}

impl<'a, 'b> std::iter::Iterator for ChunkTreeIterator<'a, 'b> {
    type Item = IterationItem<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.errored {
            return None;
        }

        let current_item = if let Some(current_item) = self.traversal_stack.pop() {
            current_item
        } else {
            return None;
        };

        let raw_chunk = self.tree.chunks[current_item.index].raw_chunk;

        let chunk = match raw_chunk.try_into() {
            Ok(v) => v,
            Err(e) => {
                self.errored = true;
                return Some(IterationItem {
                    depth: current_item.depth,
                    state: std::default::Default::default(),
                    raw_chunk,
                    chunk: Err(e),
                });
            }
        };

        let iter_item = IterationItem {
            depth: current_item.depth,
            state: std::default::Default::default(),
            raw_chunk,
            chunk: Ok(chunk),
        };

        for child in &self.tree.chunks[current_item.index].children {
            self.traversal_stack.push(TraversalItem {
                depth: current_item.depth + 1,
                index: *child,
                state: std::default::Default::default(),
            });
        }

        Some(iter_item)
    }
}
