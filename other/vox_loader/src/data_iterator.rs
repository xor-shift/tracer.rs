use color_eyre::eyre::eyre;

use super::chunk::ChunkID;

#[derive(Copy, Clone, Debug)]
pub struct RawChunk<'a> {
    pub(super) parent_chunk_index: usize,
    pub(super) ident: ChunkID,
    pub(super) data: &'a [u8],
}

pub struct ChunkDataIterator<'a> {
    consumed_chunks: usize,
    consumed_bytes: usize,
    remaining_data: &'a [u8],

    parent_stack: Vec<(usize, usize)>,

    errored: bool,
}

impl<'a> ChunkDataIterator<'a> {
    pub fn new(bytes: &'a [u8]) -> ChunkDataIterator<'a> {
        Self {
            consumed_chunks: 0,
            consumed_bytes: 0,
            remaining_data: bytes,

            parent_stack: Vec::new(),

            errored: false,
        }
    }
}

impl<'a> Iterator for ChunkDataIterator<'a> {
    type Item = color_eyre::Result<RawChunk<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.errored {
            return None;
        }

        if self.remaining_data.len() == 0 {
            return None;
        }

        if self.remaining_data.len() < 12 {
            self.errored = true;
            return Some(Err(eyre!("not enough bytes remain to form a chunk header, ({} remain, >= 12 is expected)", self.remaining_data.len())));
        }

        let header_bytes = &self.remaining_data[..12];
        self.remaining_data = &self.remaining_data[12..];

        let ident = [header_bytes[0], header_bytes[1], header_bytes[2], header_bytes[3]].map(|v| v.into());
        let ident = if let Some(v) = ident.as_ascii() {
            *v
        } else {
            self.errored = true;
            return Some(Err(eyre!("non-ascii chunk identifier {:#?}", ident)));
        };

        let chunk_len = u32::from_le_bytes((&header_bytes[4..8]).try_into().unwrap());
        let child_len = u32::from_le_bytes((&header_bytes[8..12]).try_into().unwrap());

        if self.remaining_data.len() < chunk_len as usize {
            self.errored = true;
            return Some(Err(eyre!("chunk declares more bytes than the remaining amount ({} < {})", self.remaining_data.len(), chunk_len)));
        }

        let chunk_data = &self.remaining_data[..chunk_len as usize];
        self.remaining_data = &self.remaining_data[chunk_len as usize..];

        let parent_index = if let Some(parent) = self.parent_stack.last() {
            parent.0
        } else {
            self.consumed_chunks // self
        };

        for (_idx, rem) in &mut self.parent_stack {
            *rem -= 12 + chunk_len as usize;
        }

        if let Some(parent) = self.parent_stack.last() {
            if parent.1 < child_len as usize {
                self.errored = true;
                return Some(Err(eyre!("chunk is trying to declare more child bytes than its parent permits (parent has {}B left, chunk declares {})", parent.1, child_len)));
            }
        }

        self.parent_stack.retain(|v| v.1 != 0);

        if child_len != 0 {
            self.parent_stack.push((self.consumed_chunks, child_len as usize));
        }

        self.consumed_bytes += 12 + chunk_len as usize;
        self.consumed_chunks += 1;
        Some(Ok(RawChunk {
            parent_chunk_index: parent_index,
            ident: ChunkID(ident),
            data: chunk_data,
        }))
    }
}

pub struct TreeChunk<'a> {
    pub(super) raw_chunk: RawChunk<'a>,
    pub(super) children: Vec<usize>,
}

pub struct ChunkTree<'a> {
    pub(super) chunks: Vec<TreeChunk<'a>>,
    pub(super) root_indices: Vec<usize>,
}

impl<'a> ChunkTree<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> color_eyre::Result<ChunkTree<'a>> {
        let chunks: Vec<_> = ChunkDataIterator::new(bytes).collect::<color_eyre::Result<Vec<_>, _>>()?;

        // i am leaving this in, lol
        /*
        // what in the name of hell
        if let Some(last) = chunks.last() {
            if last.is_err() {
                let mut chunks = chunks;
                chunks.pop().unwrap()?;
                unreachable!();
            }
        } else {
            return Ok(ChunkTree {
                chunks: Vec::new(),
                root_indices: Vec::new(),
            });
        }
        */

        if chunks.len() == 0 {
            return Ok(ChunkTree {
                chunks: Vec::new(),
                root_indices: Vec::new(),
            });
        }

        let mut root_indices = Vec::new();
        let mut tree_chunks: Vec<_> = chunks.into_iter().map(|c| TreeChunk { raw_chunk: c, children: Vec::new() }).collect();

        // eugh
        for i in 0..tree_chunks.len() {
            let parent = tree_chunks[i].raw_chunk.parent_chunk_index;
            if parent == i {
                root_indices.push(i);
                continue;
            }

            tree_chunks[parent].children.push(i);
        }

        Ok(Self { chunks: tree_chunks, root_indices })
    }
}

#[cfg(test)]
mod test {
    use super::super::state::ChunkTreeIterator;

    use super::*;

    #[test]
    fn test_iter() {
        let file = include_bytes!("../../../run/vox/anim/deer.vox");

        let header_bytes = &file[..8];
        let file = &file[8..];

        let tree = ChunkTree::from_bytes(file);

        let tree = match tree {
            Ok(v) => v,
            Err(e) => panic!("could not create a tree, error: {}", e),
        };

        for v in ChunkTreeIterator::new(&tree) {
            let chunk = match v.chunk {
                Ok(v) => v,
                Err(e) => panic!("a chunk with ident {:?} at depth {} is invalid: {e}", v.raw_chunk.ident, v.depth),
            };

            println!("{:?} @ depth {}", v.raw_chunk.ident, v.depth);
        }
    }
}
