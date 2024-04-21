use super::extent::VoxelExtent;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(super) enum Turn {
    NNN = 0, // -x, -y, -z
    PNN = 1, // +x, -y, -z
    NPN = 2,
    PPN = 3,
    NNP = 4,
    PNP = 5,
    NPP = 6,
    PPP = 7,
}

// why did i bother honestly
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct NodePath<const C: usize = 8>
where
    [Turn; C * 3]:,
{
    //pub(super) data: [u8; C],
    pub(super) data: [Turn; C * 3],
    pub(super) path_length: usize,
}

const fn depth_bits_for_depth(depth: usize) -> usize { (if depth.is_power_of_two() { depth.trailing_zeros() + 1 } else { depth.next_power_of_two().trailing_zeros() }) as usize }
const fn pack_bits_for_depth(depth: usize) -> usize { 3 * depth + depth_bits_for_depth(depth) }

const fn max_depth_for_pack(bits: usize) -> usize {
    let mut last_good_depth = 0;

    loop {
        let next_depth = last_good_depth + 1;
        let bits_for_next = pack_bits_for_depth(next_depth);
        if bits_for_next > bits {
            break;
        }

        last_good_depth = next_depth;
    }

    last_good_depth
}

// bit_arr is little-endian
const fn extract_bits<const COUNT: usize>(bit_arr: &[u8], bit_idx: usize) -> u8 {
    assert!(COUNT <= 8);

    let start_byte = bit_idx / 8;
    let bit_index_into_bytes = bit_idx % 8;

    if bit_index_into_bytes + COUNT <= 8 {
        let byte = bit_arr[start_byte];

        let left = 8 - (COUNT + bit_index_into_bytes);
        let right = bit_index_into_bytes;

        // xxxxxxxx xx111xxx
        (byte << left) >> (right + left)
    } else {
        let bytes = [bit_arr[start_byte], bit_arr[start_byte + 1]];

        // xxxxxx11 1xxxxxxx
        let lower = bytes[0] >> bit_index_into_bytes;

        // xxxxxx11 -> 00000110
        let upper_bitct = COUNT - (8 - bit_index_into_bytes);
        let upper = (bytes[1] << (8 - upper_bitct)) >> ((8 - upper_bitct) - (COUNT - upper_bitct));

        upper | lower
    }
}

const fn inject_bits<const COUNT: usize>(bit_arr: &[u8], bit_idx: usize, val: u8) {
    todo!();
}

impl<const C: usize> NodePath<C>
where
    [Turn; C * 3]:,
{
    pub fn new() -> NodePath<C> { Self { data: [Turn::NNN; C * 3], path_length: 0 } }

    //fn max_depth() -> usize { C * 8 / 3 }
    fn max_depth() -> usize { C * 3 }

    /// packs path for use in WGSL
    pub fn pack<const M: usize>(&self) -> [u32; M] {
        let mut ret = [0; M];

        for i in 0..self.depth() {
            let bit_idx = (i * 4) % 32;
            let word_idx = (i * 4) / 32;

            let v = self.nth_turn(i) as u8 | 0b1000;
            ret[word_idx] |= (v as u32) << bit_idx;
        }

        ret
    }

    pub fn from_pack<const M: usize>(pack: [u32; M]) -> NodePath<C> {
        let mut ret = NodePath::new();

        for i in 0.. {
            let bit_idx = (i * 4) % 32;
            let word_idx = (i * 4) / 32;

            let v = ((pack[word_idx] >> bit_idx) & 0xF) as u8;
            if v & 0x8 == 0 {
                break;
            }

            ret.take_turn(unsafe { std::mem::transmute(v & 0x7) });
        }

        ret
    }

    pub fn depth(&self) -> usize { self.path_length as usize }

    pub fn nth_turn(&self, n: usize) -> Turn {
        /*if n >= self.depth() {
            panic!("access is out of bounds (n = {n}, depth = {})", self.depth());
        }

        unsafe { std::mem::transmute(extract_bits::<3>(&self.data, n * 3)) }*/
        self.data[n]
    }

    pub fn last_turn(&self) -> Option<Turn> {
        if self.depth() == 0 {
            None
        } else {
            Some(self.nth_turn(self.depth() - 1))
        }
    }

    pub fn take_turn(&mut self, turn: Turn) {
        self.data[self.path_length] = turn;
        self.path_length += 1;
    }

    pub fn after_taken_turn(&self, turn: Turn) -> NodePath<C> {
        let mut ret = *self;
        ret.take_turn(turn);
        ret
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bit_extraction() {
        let bits = [
            0b11000110, //
            0b10000011, //
            0b00011111, //
            0b11110000, //
            0b00001111, //
            0b11100000, //
            0b01111111, //
        ];
        assert_eq!(extract_bits::<1>(&bits, 0), 0b0);
        assert_eq!(extract_bits::<2>(&bits, 1), 0b11);
        assert_eq!(extract_bits::<3>(&bits, 3), 0b000);
        assert_eq!(extract_bits::<4>(&bits, 6), 0b1111);
        assert_eq!(extract_bits::<5>(&bits, 10), 0b00000);
        assert_eq!(extract_bits::<6>(&bits, 15), 0b111111);
        assert_eq!(extract_bits::<7>(&bits, 21), 0b0000000);
        assert_eq!(extract_bits::<8>(&bits, 28), 0b11111111);
    }

    #[test]
    fn test_path_pack_bitlens() {
        let expected = [0, 4, 8, 11, 15, 18, 21, 24];
        for (depth, bits) in expected.into_iter().enumerate() {
            assert_eq!(bits, pack_bits_for_depth(depth));
        }

        assert_eq!(0, max_depth_for_pack(0));
        assert_eq!(0, max_depth_for_pack(1));
        assert_eq!(0, max_depth_for_pack(2));
        assert_eq!(0, max_depth_for_pack(3));
        assert_eq!(1, max_depth_for_pack(4));
    }

    #[test]
    fn test_path_packing() {
        let mut path = NodePath::<8>::new();
        assert_eq!(path, NodePath::from_pack(path.pack::<2>()));

        path.take_turn(Turn::NNN);
        assert_eq!(path, NodePath::from_pack(path.pack::<2>()));

        path.take_turn(Turn::NNP);
        assert_eq!(path, NodePath::from_pack(path.pack::<2>()));

        path.take_turn(Turn::PPP);
        assert_eq!(path, NodePath::from_pack(path.pack::<2>()));
    }
}
