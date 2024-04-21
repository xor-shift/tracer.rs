use super::{
    chunk::Chunk,
    data_iterator::{ChunkTree, RawChunk},
    transform::Transform,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct State {
    pub origin: [i32; 3],
    pub transform: Transform,
    pub size: [u32; 3],
}

impl std::default::Default for State {
    fn default() -> Self {
        Self {
            origin: [0, 0, 0],
            transform: Transform::identity(),
            size: [0, 0, 0],
        }
    }
}

#[derive(Clone, Copy)]
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
                    state: current_item.state,
                    raw_chunk,
                    chunk: Err(e),
                });
            }
        };

        let mut new_state = current_item.state;

        match chunk {
            Chunk::Size { size } => new_state.size = size,
            _ => (),
        }

        /*
        a
         b
         c
          d
          e
           x
         f
         g

        a    0
        gfcb 1111
        gfc  111
        gfed 1122
        gfe  112
        gfx  113
        gf   11
        g    1

        conclusion: to propagate state changes sideways, just go back in the stack and update items of equal depth
        */
        for i in 0..self.traversal_stack.len() {
            if self.traversal_stack[i].depth != current_item.depth {
                break;
            }

            self.traversal_stack[i].state = new_state;
        }

        let iter_item = IterationItem {
            depth: current_item.depth,
            state: new_state,
            raw_chunk,
            chunk: Ok(chunk),
        };

        for child in self.tree.chunks[current_item.index].children.iter().rev() {
            self.traversal_stack.push(TraversalItem {
                depth: current_item.depth + 1,
                index: *child,
                state: new_state,
            });
        }

        Some(iter_item)
    }
}
