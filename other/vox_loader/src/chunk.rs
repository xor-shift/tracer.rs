use color_eyre::eyre::{ensure, eyre};

use super::data_iterator::RawChunk;

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct ChunkID(pub [std::ascii::Char; 4]);

impl ChunkID {
    pub const fn from_str(s: &str) -> ChunkID {
        let chars = if let Some(chars) = s.as_ascii() {
            chars
        } else {
            panic!("ChunkID::from_str expects ascii characters");
        };

        match chars {
            [a, b, c, d] => ChunkID([*a, *b, *c, *d]),
            _ => panic!("ChunkID::from_str expects 4 ascii characters"),
        }
    }

    pub fn as_str(&self) -> &str { self.0.as_str() }
}

impl std::fmt::Debug for ChunkID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_fmt(format_args!("{}", self.as_str())) }
}

#[allow(dead_code)]
pub mod chunk_ids {
    use super::ChunkID;

    pub const MAIN: ChunkID = ChunkID::from_str("MAIN");
    pub const SIZE: ChunkID = ChunkID::from_str("SIZE");
    pub const XYZI: ChunkID = ChunkID::from_str("XYZI");
    pub const RGBA: ChunkID = ChunkID::from_str("RGBA");
    pub const MATL: ChunkID = ChunkID::from_str("MATL");
    pub const NTRN: ChunkID = ChunkID::from_str("nTRN");
    pub const NGRP: ChunkID = ChunkID::from_str("nGRP");
    pub const NSHP: ChunkID = ChunkID::from_str("nSHP");
    pub const LAYR: ChunkID = ChunkID::from_str("LAYR");
    pub const ROBJ: ChunkID = ChunkID::from_str("rOBJ");
    pub const RCAM: ChunkID = ChunkID::from_str("rCAM");
    pub const NOTE: ChunkID = ChunkID::from_str("NOTE");
    pub const IMAP: ChunkID = ChunkID::from_str("IMAP");
}

unsafe fn indexing_utility<'a, const STRIDE: usize, T: Sized>(data: &'a [T], strideless_index: usize) -> &'a [T; STRIDE] {
    let ptr = data.as_ptr() as *const T;
    let data_ptr = ptr.add(strideless_index * STRIDE);
    let dyn_slice = std::slice::from_raw_parts::<'a, _>(data_ptr, STRIDE);
    &*(dyn_slice.as_ptr() as *const [T; STRIDE])
}

pub struct XYZIIterator<'a> {
    chunk: XYZIChunk<'a>,
    index: usize,
}

impl<'a> std::iter::Iterator for XYZIIterator<'a> {
    type Item = <XYZIChunk<'a> as std::ops::Index<usize>>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.chunk.len() {
            return None;
        }

        let ret = self.chunk[self.index];
        self.index += 1;

        Some(ret)
    }
}

#[derive(Clone, Copy)]
pub struct XYZIChunk<'a> {
    raw_data: &'a [u8], // length == 0 mod 4
}

impl<'a> XYZIChunk<'a> {
    pub const fn len(&self) -> usize { self.raw_data.len() / 4 }

    pub fn iter(&self) -> XYZIIterator<'a> { XYZIIterator { chunk: *self, index: 0 } }
}

impl<'a> std::convert::TryFrom<RawChunk<'a>> for XYZIChunk<'a> {
    type Error = color_eyre::Report;

    fn try_from(value: RawChunk<'a>) -> Result<Self, Self::Error> {
        ensure!(value.data.len() >= 4, "the raw chunk for an XYZI chunk contains insufficient data (expected at least 4, got {})", value.data.len());
        let voxel_data = &value.data[4..];

        let num_voxels = u32::from_le_bytes(unsafe { *(value.data.as_ptr() as *const [u8; 4]) }) as usize;

        ensure!(
            voxel_data.len() == num_voxels * 4,
            "the raw chunk for an XYZI chunk contains insufficient data for the payload (expected {} bytes, got {})",
            num_voxels * 4,
            voxel_data.len()
        );

        Ok(Self { raw_data: voxel_data })
    }
}

impl<'a> std::ops::Index<usize> for XYZIChunk<'a> {
    type Output = [u8; 4];

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.raw_data.len() / 4, "out-of-bounds index (given: {}, len: {})", index, self.raw_data.len() / 4);

        unsafe { indexing_utility(self.raw_data, index) }
    }
}

pub struct RGBAChunk<'a> {
    raw_data: &'a [u8; 256 * 4],
}

impl<'a> RGBAChunk<'a> {
    pub const fn len(&self) -> usize { 256 }
}

impl<'a> std::convert::TryFrom<RawChunk<'a>> for RGBAChunk<'a> {
    type Error = color_eyre::Report;

    fn try_from(value: RawChunk<'a>) -> Result<Self, Self::Error> {
        ensure!(value.data.len() == 256 * 4, "the raw chunk for an RGBA chunk contains an inadequate number of bytes (expected 1024, got {})", value.data.len());

        Ok(Self {
            raw_data: unsafe { &*(value.data.as_ptr() as *const [u8; 256 * 4]) },
        })
    }
}

impl<'a> std::ops::Index<usize> for RGBAChunk<'a> {
    type Output = [u8; 4];

    fn index(&self, index: usize) -> &'a Self::Output {
        assert!(index < 256, "out-of-bounds index (an RGBA palette contains 256 entries)");

        unsafe { indexing_utility(self.raw_data, index) }
    }
}

pub enum Chunk<'a> {
    Unknown(&'a [u8]),
    Main,
    Size { size: [u32; 3] },
    Palette(RGBAChunk<'a>),
    Voxels(XYZIChunk<'a>),
}

impl<'a> Chunk<'a> {}

impl<'a> std::convert::TryFrom<RawChunk<'a>> for Chunk<'a> {
    type Error = color_eyre::Report;

    fn try_from(value: RawChunk<'a>) -> Result<Self, Self::Error> {
        match value.ident {
            chunk_ids::MAIN => {
                if value.data.is_empty() {
                    Ok(Self::Main)
                } else {
                    Err(eyre!("the MAIN chunk should be empty"))
                }
            }

            chunk_ids::SIZE => {
                ensure!(value.data.len() == 12, "a SIZE chunk should have exactly 12 bytes (got {})", value.data.len());

                let x = u32::from_le_bytes((&value.data[0..4]).try_into().unwrap());
                let y = u32::from_le_bytes((&value.data[4..8]).try_into().unwrap());
                let z = u32::from_le_bytes((&value.data[8..12]).try_into().unwrap());

                Ok(Self::Size { size: [x, y, z] })
            }

            chunk_ids::RGBA => Ok(Self::Palette(RGBAChunk::try_from(value)?)),
            chunk_ids::XYZI => Ok(Self::Voxels(XYZIChunk::try_from(value)?)),

            _ => Ok(Self::Unknown(value.data)),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_xyzi_indexing() {
        let raw_chunk = RawChunk {
            ident: chunk_ids::XYZI,
            parent_chunk_index: 0,
            data: &[2, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4],
        };

        let chunk = XYZIChunk::try_from(raw_chunk);

        assert!(chunk.is_ok());

        let chunk = chunk.unwrap();

        assert_eq!(chunk[0], [0, 0, 0, 0]);
        assert_eq!(chunk[1], [1, 2, 3, 4]);
    }

    #[test]
    fn test_rgba_indexing() {
        let raw_chunk = RawChunk {
            ident: chunk_ids::RGBA,
            parent_chunk_index: 0,
            #[rustfmt::skip]
            data: &[
                // 64 per row, 256 per group
                0,  1,  2,  3,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                4,  5,  6,  7,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                8,  9,  10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0,  1,  2,  3,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                4,  5,  6,  7,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                8,  9,  10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0,  1,  2,  3,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                4,  5,  6,  7,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                8,  9,  10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0,  1,  2,  3,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                4,  5,  6,  7,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                8,  9,  10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        };

        let chunk = RGBAChunk::try_from(raw_chunk);

        assert!(chunk.is_ok());

        let chunk = chunk.unwrap();

        assert_eq!(chunk[0], [0, 1, 2, 3]);
        assert_eq!(chunk[1], [0, 0, 0, 0]);
        assert_eq!(chunk[16], [4, 5, 6, 7]);
    }
}
