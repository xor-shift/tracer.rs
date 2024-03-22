#![feature(ascii_char)]

mod chunk;
mod data_iterator;
mod state;
mod transform;

use color_eyre::eyre::{ensure, eyre};

use self::{data_iterator::ChunkTree, state::ChunkTreeIterator};

pub struct File<'a> {
    pub version: u32,
    tree: ChunkTree<'a>,
}

impl<'a> File<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> color_eyre::Result<File<'a>> {
        ensure!(bytes.len() >= 8, "not enough bytes to form a file header");

        let header_bytes = &bytes[..8];

        match &header_bytes[..4] {
            [0x56, 0x4F, 0x58, 0x20] => {}
            _ => return Err(eyre!("mad magic value")),
        };

        let version = u32::from_le_bytes((&header_bytes[4..]).try_into().unwrap());

        let chunk_bytes = &bytes[8..];

        let tree = ChunkTree::from_bytes(chunk_bytes)?;

        Ok(Self { version, tree })
    }

    pub fn iter(&self) -> ChunkTreeIterator<'a, '_> { ChunkTreeIterator::new(&self.tree) }
}
