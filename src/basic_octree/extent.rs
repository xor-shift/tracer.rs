use super::path::{Path, Turn};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct DiscreteExtent {
    pub(super) min: cgmath::Point3<i64>,
    pub(super) max: cgmath::Point3<i64>,
}

impl DiscreteExtent {
    // the point closest to +inf on the NNN split
    pub(super) fn lower_center(&self) -> cgmath::Point3<i64> { cgmath::point3(i64::midpoint(self.min.x, self.max.x), i64::midpoint(self.min.y, self.max.y), i64::midpoint(self.min.z, self.max.z)) }

    // the point closest to -inf on the PPP split
    pub(super) fn upper_center(&self) -> cgmath::Point3<i64> { self.lower_center().zip(cgmath::point3(1, 1, 1), i64::wrapping_add) }

    pub(super) fn partition<T: Sized>(&self, data: &mut [(cgmath::Point3<i64>, T)]) -> [usize; 7] {
        /*
            --x-
            abcd |
            efgh |
            ijkl y
            mnop |

            stored: acikpnhfjgbemold
            partition x: ainfjbem||ckphgold   1->2
            partition y: afbe|injm||chgd|kpol 2->4
        */

        let center_pt = self.lower_center();

        let z_split = data.iter_mut().partition_in_place(|(point, _mat)| point.z <= center_pt.z);

        let y_split_0 = (&mut data[..z_split]).iter_mut().partition_in_place(|(point, _mat)| point.y <= center_pt.y);
        let y_split_1 = (&mut data[z_split..]).iter_mut().partition_in_place(|(point, _mat)| point.y <= center_pt.y) + z_split;

        let x_split_0 = (&mut data[..y_split_0]).iter_mut().partition_in_place(|(point, _mat)| point.x <= center_pt.x);
        let x_split_1 = (&mut data[y_split_0..z_split]).iter_mut().partition_in_place(|(point, _mat)| point.x <= center_pt.x) + y_split_0;
        let x_split_2 = (&mut data[z_split..y_split_1]).iter_mut().partition_in_place(|(point, _mat)| point.x <= center_pt.x) + z_split;
        let x_split_3 = (&mut data[y_split_1..]).iter_mut().partition_in_place(|(point, _mat)| point.x <= center_pt.x) + y_split_1;

        [x_split_0, y_split_0, x_split_1, z_split, x_split_2, y_split_1, x_split_3]
    }

    /// midpoints are considered to be in the negative half of whatever axis they belong to:
    /// ```js
    /// for a 5x5x1 grid:
    ///
    /// 22233
    /// 22233
    /// 00011
    /// 00011
    /// 00011
    /// ```
    ///
    /// split indices are consistent with self.partition:
    /// ```js
    /// -x, -y, -z
    /// +x, -y, -z
    /// -x, +y, -z
    /// +x, +y, -z
    /// -x, -y, +z
    /// +x, -y, +z
    /// -x, +y, +z
    /// +x, +y, +z
    ///
    ///         +----+----+
    ///        /6   /7   /|
    ///       /    /    / |
    ///      +----+----+7 +
    ///     /2   /3   /| /|
    ///    /    /    / |/ |
    ///   +----+----+3 +5 +
    ///   |2   |3   | /| /
    ///   |    |    |/ |/
    /// y +----+----+1 +  z
    /// + |0   |1   | /  +
    /// ^ |    |    |/  ^
    /// | +----+----+  /
    ///  -> +x
    /// ```
    pub(super) fn split_for_turn(&self, turn: Turn) -> DiscreteExtent {
        let selections = [(turn as usize >> 0) & 1, (turn as usize >> 1) & 1, (turn as usize >> 2) & 1];

        let lo_center = self.lower_center();
        let hi_center = self.upper_center();

        return Self {
            min: cgmath::point3([self.min, hi_center][selections[0]].x, [self.min, hi_center][selections[1]].y, [self.min, hi_center][selections[2]].z),
            max: cgmath::point3([lo_center, self.max][selections[0]].x, [lo_center, self.max][selections[1]].y, [lo_center, self.max][selections[2]].z),
        };
    }

    pub(super) fn figure_out_path<const CAP: usize>(&self, point: cgmath::Point3<i64>) -> Path<CAP>
    where
        [(); (CAP + 7) / 8]:,
    {
        let mut ret = Path::new();

        let mut cur = *self;

        while cur.min.zip(cur.max, |a, b| a < b) != cgmath::point3(false, false, false) {
            let center = cur.lower_center();
            let negative = point.zip(center, |a, b| a <= b);

            let weights = negative.map(|v| !v as u8).zip(cgmath::point3(1, 2, 4), u8::wrapping_mul);
            let turn = weights.x + weights.y + weights.z;

            let turn = unsafe { std::mem::transmute(turn) };

            ret.take_turn(turn);
            cur = cur.split_for_turn(turn);
        }

        ret
    }
}

impl From<(cgmath::Point3<i64>, cgmath::Point3<i64>)> for DiscreteExtent {
    fn from(value: (cgmath::Point3<i64>, cgmath::Point3<i64>)) -> Self { Self { min: value.0, max: value.1 } }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ve_partitioning_0() {
        #[rustfmt::skip]
        let original_values = vec![
            (cgmath::point3(0, 0, 0), 'a'),
            (cgmath::point3(1, 0, 0), 'b'),
            (cgmath::point3(0, 1, 0), 'c'),
            (cgmath::point3(1, 1, 0), 'd'),
            (cgmath::point3(0, 0, 1), 'e'),
            (cgmath::point3(1, 0, 1), 'f'),
            (cgmath::point3(0, 1, 1), 'g'),
            (cgmath::point3(1, 1, 1), 'h'),
        ];

        let mut values = original_values.clone();
        values.reverse();

        let bound = DiscreteExtent {
            min: cgmath::Point3 { x: 0, y: 0, z: 0 },
            max: cgmath::Point3 { x: 1, y: 1, z: 1 },
        };

        let splits = bound.partition(values.as_mut_slice());

        assert_eq!(values, original_values);
        assert_eq!(splits, [1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_ve_partitioning_1() {
        #[rustfmt::skip]
        let original_values = vec![
            (cgmath::point3(0, 0, 0), 'a'),
            (cgmath::point3(1, 0, 0), 'b'),
            (cgmath::point3(0, 1, 0), 'c'),
            (cgmath::point3(1, 1, 0), 'd'),
        ];

        let mut values = original_values.clone();
        values.reverse();

        let bound = DiscreteExtent {
            min: cgmath::Point3 { x: 0, y: 0, z: 0 },
            max: cgmath::Point3 { x: 1, y: 1, z: 0 },
        };

        let splits = bound.partition(values.as_mut_slice());

        assert_eq!(values, original_values);
        assert_eq!(splits, [1, 2, 3, 4, 4, 4, 4]);
    }

    #[test]
    fn test_ve_midpoint() {
        #[rustfmt::skip]
        let original_values = vec![
            (cgmath::point3(0, 0, 0), '0'),
            (cgmath::point3(1, 0, 0), '0'),
            (cgmath::point3(2, 0, 0), '0'),
            (cgmath::point3(3, 0, 0), '1'),
            (cgmath::point3(4, 0, 0), '1'),

            (cgmath::point3(0, 1, 0), '0'),
            (cgmath::point3(1, 1, 0), '0'),
            (cgmath::point3(2, 1, 0), '0'),
            (cgmath::point3(3, 1, 0), '1'),
            (cgmath::point3(4, 1, 0), '1'),

            (cgmath::point3(0, 2, 0), '0'),
            (cgmath::point3(1, 2, 0), '0'),
            (cgmath::point3(2, 2, 0), '0'),
            (cgmath::point3(3, 2, 0), '1'),
            (cgmath::point3(4, 2, 0), '1'),

            (cgmath::point3(0, 3, 0), '2'),
            (cgmath::point3(1, 3, 0), '2'),
            (cgmath::point3(2, 3, 0), '2'),
            (cgmath::point3(3, 3, 0), '3'),
            (cgmath::point3(4, 3, 0), '3'),

            (cgmath::point3(0, 4, 0), '2'),
            (cgmath::point3(1, 4, 0), '2'),
            (cgmath::point3(2, 4, 0), '2'),
            (cgmath::point3(3, 4, 0), '3'),
            (cgmath::point3(4, 4, 0), '3'),
        ];

        let mut values = original_values.clone();
        values.reverse();

        let bound = DiscreteExtent {
            min: cgmath::Point3 { x: 0, y: 0, z: 0 },
            max: cgmath::Point3 { x: 4, y: 4, z: 0 },
        };

        let splits = bound.partition(values.as_mut_slice());

        assert_eq!(splits, [9, 15, 21, 25, 25, 25, 25]);

        assert_eq!(&(&values[0..splits[0]]).iter().map(|v| v.1).collect::<Vec<_>>(), &['0', '0', '0', '0', '0', '0', '0', '0', '0']);
        assert_eq!(&(&values[splits[0]..splits[1]]).iter().map(|v| v.1).collect::<Vec<_>>(), &['1', '1', '1', '1', '1', '1']);
        assert_eq!(&(&values[splits[1]..splits[2]]).iter().map(|v| v.1).collect::<Vec<_>>(), &['2', '2', '2', '2', '2', '2']);
        assert_eq!(&(&values[splits[2]..splits[3]]).iter().map(|v| v.1).collect::<Vec<_>>(), &['3', '3', '3', '3']);
    }

    #[test]
    fn path_from_extent() {
        let extent = DiscreteExtent {
            min: cgmath::point3(0, 0, 0),
            max: cgmath::point3(7, 7, 7),
        };

        assert_eq!(extent.figure_out_path(cgmath::point3(1, 2, 3)), Path::<24>::from_digits(&[0, 6, 5]));
    }
}
