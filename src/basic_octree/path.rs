use std::fmt::Write;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Turn {
    NNN = 0, // -x, -y, -z
    PNN = 1, // +x, -y, -z
    NPN = 2, // -x, +y, -z
    PPN = 3, // +x, +y, -z
    NNP = 4, // -x, -y, +z
    PNP = 5, // +x, -y, +z
    NPP = 6, // -x, +y, +z
    PPP = 7, // +x, +y, +z
}

impl std::fmt::Debug for Turn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Turn::NNN => f.write_str("NNN"),
            Turn::PNN => f.write_str("PNN"),
            Turn::NPN => f.write_str("NPN"),
            Turn::PPN => f.write_str("PPN"),
            Turn::NNP => f.write_str("NNP"),
            Turn::PNP => f.write_str("PNP"),
            Turn::NPP => f.write_str("NPP"),
            Turn::PPP => f.write_str("PPP"),
        }
    }
}

impl From<usize> for Turn {
    fn from(value: usize) -> Self {
        assert!(value < 8);
        unsafe { std::mem::transmute(value as u8) }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Path<const CAP: usize = 24>
where
    [u32; (CAP + 7) / 8]:,
{
    data: [u32; (CAP + 7) / 8],
    length_memo: usize,
}

impl<const CAP: usize> Path<CAP>
where
    [u32; (CAP + 7) / 8]:,
{
    pub const fn new() -> Path<CAP> { Path { data: [0; (CAP + 7) / 8], length_memo: 0 } }

    pub const fn from_turns(turns: &[Turn]) -> Path<CAP> {
        let mut ret = Path::new();

        let mut i = 0;
        loop {
            if i >= turns.len() {
                break;
            }

            ret.take_turn(turns[i]);

            i += 1;
        }

        ret
    }

    pub const fn from_digits(turns: &[u8]) -> Path<CAP> {
        let mut ret = Path::new();

        let mut i = 0;
        loop {
            if i >= turns.len() {
                break;
            }

            ret.take_turn(unsafe { std::mem::transmute(turns[i]) });

            i += 1;
        }

        ret
    }

    pub fn from_data(data: &[u32; (CAP + 7) / 8]) -> Path<CAP> {
        let length = data //
            .into_iter()
            .enumerate()
            .find(|(_i, v)| **v < 0x8000_0000)
            .map(|(end_word_idx, _v)| {
                let end_word = data[end_word_idx];
                let subtract = end_word.leading_zeros() / 4;
                end_word_idx * 8 + 8 - subtract as usize
            })
            .unwrap_or(CAP);

        Self { data: *data, length_memo: length }
    }

    pub const fn take_turn(&mut self, turn: Turn) {
        let byte_idx = self.length_memo / 8;
        let bit_idx = (self.length_memo % 8) * 4;

        self.length_memo += 1;
        self.data[byte_idx] |= ((turn as u8 | 0x8) as u32) << bit_idx;
    }

    pub const fn after_taken_turn(mut self, turn: Turn) -> Self {
        self.take_turn(turn);
        self
    }

    /// - lowest bit is the part of the first turn (the turn closest to the root)
    /// - lowest word contains the first turns
    /// - each turn is encoded with 4 bits, highest of which is always 1
    pub const fn data(&self) -> &[u32; (CAP + 7) / 8] { &self.data }

    pub const fn depth(&self) -> usize { self.length_memo }

    pub const fn nth_turn(&self, n: usize) -> Turn {
        assert!(n < self.depth(), "n must be in bounds");

        let byte_idx = n / 8;
        let bit_idx = (n % 8) * 4;

        let value = (self.data[byte_idx] >> bit_idx) & 0x7;

        unsafe { std::mem::transmute(value as u8) }
    }

    pub fn truncating_cap_cast<const OTHER_CAP: usize>(&self) -> Path<OTHER_CAP>
    where
        [u32; (OTHER_CAP + 7) / 8]:,
    {
        let mut ret = Path::new();

        for i in 0..self.depth().min(OTHER_CAP) {
            ret.take_turn(self.nth_turn(i));
        }

        ret
    }
}

impl<const CAP: usize> IntoIterator for Path<CAP>
where
    [u32; (CAP + 7) / 8]:,
{
    type IntoIter = PathIterator<CAP>;
    type Item = Turn;

    fn into_iter(self) -> Self::IntoIter { PathIterator { path: self, i: 0 } }
}

pub struct PathIterator<const CAP: usize>
where
    [u32; (CAP + 7) / 8]:,
{
    path: Path<CAP>,
    i: usize,
}

impl<const CAP: usize> Iterator for PathIterator<CAP>
where
    [u32; (CAP + 7) / 8]:,
{
    type Item = Turn;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i == self.path.depth() {
            None
        } else {
            self.i += 1;
            Some(self.path.nth_turn(self.i - 1))
        }
    }
}

impl<const CAP: usize> std::fmt::Debug for Path<CAP>
where
    [u32; (CAP + 7) / 8]:,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('[')?;

        for (i, turn) in self.into_iter().enumerate() {
            f.write_fmt(format_args!("{}{:?}", if i == 0 { "" } else { ", " }, turn))?;
        }

        f.write_char(']')
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
        let mut path: Path<9> = Path::new();
        assert_eq!(path.data(), &[0, 0]);

        path.take_turn(Turn::NNN);
        assert_eq!(path.data(), &[0x00000008, 0x0000000]);
        assert_eq!(path, Path::from_data(path.data()));

        path.take_turn(Turn::PNN);
        assert_eq!(path.data(), &[0x00000098, 0x0000000]);
        assert_eq!(path, Path::from_data(path.data()));

        path.take_turn(Turn::PPN);
        assert_eq!(path.data(), &[0x00000B98, 0x0000000]);
        assert_eq!(path, Path::from_data(path.data()));

        path.take_turn(Turn::NPP);
        assert_eq!(path.data(), &[0x0000EB98, 0x0000000]);
        assert_eq!(path, Path::from_data(path.data()));

        path.take_turn(Turn::PPP);
        assert_eq!(path.data(), &[0x000FEB98, 0x0000000]);
        assert_eq!(path, Path::from_data(path.data()));

        path.take_turn(Turn::PPP);
        assert_eq!(path.data(), &[0x00FFEB98, 0x0000000]);
        assert_eq!(path, Path::from_data(path.data()));

        path.take_turn(Turn::PPP);
        assert_eq!(path.data(), &[0x0FFFEB98, 0x0000000]);
        assert_eq!(path, Path::from_data(path.data()));

        path.take_turn(Turn::PPP);
        assert_eq!(path.data(), &[0xFFFFEB98, 0x0000000]);
        assert_eq!(path, Path::from_data(path.data()));

        path.take_turn(Turn::PPP);
        assert_eq!(path.data(), &[0xFFFFEB98, 0x000000F]);
        assert_eq!(path, Path::from_data(path.data()));
    }
}
