#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct VoxelExtent {
    pub(super) min: cgmath::Point3<i64>,
    pub(super) max: cgmath::Point3<i64>,
}

impl VoxelExtent {
    pub(super) fn center(&self) -> cgmath::Point3<i64> { cgmath::point3(i64::midpoint(self.min.x, self.max.x), i64::midpoint(self.min.y, self.max.y), i64::midpoint(self.min.z, self.max.z)) }

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

        let center_pt = self.center();

        let z_split = data.iter_mut().partition_in_place(|(point, _mat)| point.z <= center_pt.z); 

        let y_split_0 = (&mut data[..z_split]).iter_mut().partition_in_place(|(point, _mat)| point.y <= center_pt.y);
        let y_split_1 = (&mut data[z_split..]).iter_mut().partition_in_place(|(point, _mat)| point.y <= center_pt.y) + z_split;

        let x_split_0 = (&mut data[..y_split_0]).iter_mut().partition_in_place(|(point, _mat)| point.x <= center_pt.x);
        let x_split_1 = (&mut data[y_split_0..z_split]).iter_mut().partition_in_place(|(point, _mat)| point.x <= center_pt.x) + y_split_0;
        let x_split_2 = (&mut data[z_split..y_split_1]).iter_mut().partition_in_place(|(point, _mat)| point.x <= center_pt.x) + z_split;
        let x_split_3 = (&mut data[y_split_1..]).iter_mut().partition_in_place(|(point, _mat)| point.x <= center_pt.x) + y_split_1;

        [x_split_0, y_split_0, x_split_1, z_split, x_split_2, y_split_1, x_split_3]
    }

    /// split indices are consistent with self.partition:
    /// ```
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
    pub(super) fn get_split(&self, index: usize) -> VoxelExtent {
        let center_pt = self.center();

        let selections = [(index >> 0) & 1, (index >> 1) & 1, (index >> 2) & 1];

        return Self {
            min: cgmath::point3([self.min, center_pt][selections[0]].x, [self.min, center_pt][selections[1]].y, [self.min, center_pt][selections[2]].z),
            max: cgmath::point3([center_pt, self.max][selections[0]].x, [center_pt, self.max][selections[1]].y, [center_pt, self.max][selections[2]].z),
        };
    }
}

impl From<(cgmath::Point3<i64>, cgmath::Point3<i64>)> for VoxelExtent {
    fn from(value: (cgmath::Point3<i64>, cgmath::Point3<i64>)) -> Self { Self { min: value.0, max: value.1 } }
}

#[cfg(test)]
mod test {
    use super::VoxelExtent;

    #[test]
    fn test_ve_partitioning_0() {
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

        let bound = VoxelExtent {
            min: cgmath::Point3 { x: 0, y: 0, z: 0 },
            max: cgmath::Point3 { x: 1, y: 1, z: 1 },
        };

        let splits = bound.partition(values.as_mut_slice());

        assert_eq!(values, original_values);
        assert_eq!(splits, [1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_ve_partitioning_1() {
        let original_values = vec![
            (cgmath::point3(0, 0, 0), 'a'),
            (cgmath::point3(1, 0, 0), 'b'),
            (cgmath::point3(0, 1, 0), 'c'),
            (cgmath::point3(1, 1, 0), 'd'),
        ];

        let mut values = original_values.clone();
        values.reverse();

        let bound = VoxelExtent {
            min: cgmath::Point3 { x: 0, y: 0, z: 0 },
            max: cgmath::Point3 { x: 1, y: 1, z: 0 },
        };

        let splits = bound.partition(values.as_mut_slice());

        assert_eq!(values, original_values);
        assert_eq!(splits, [1, 2, 3, 4, 4, 4, 4]);
    }
}
