#[derive(Copy, Clone, PartialEq, Eq)]
pub enum MixType {
    //           1  0      0   1
    XYZ = 0b0000_01_00, // 100 010 001
    YXZ = 0b0000_00_01, // 010 100 001
    XZY = 0b0000_10_00, // 100 001 010
    YZX = 0b0000_10_01, // 010 001 100
    ZXY = 0b0000_00_10, // 001 100 010
    ZYX = 0b0000_01_10, // 001 010 100
}

impl MixType {
    pub fn apply<T: Sized + Copy>(&self, coords: [T; 3]) -> [T; 3] {
        match self {
            MixType::XYZ => [coords[0], coords[1], coords[2]],
            MixType::YXZ => [coords[1], coords[0], coords[2]],
            MixType::XZY => [coords[0], coords[2], coords[1]],
            MixType::YZX => [coords[1], coords[2], coords[0]],
            MixType::ZXY => [coords[2], coords[0], coords[1]],
            MixType::ZYX => [coords[2], coords[1], coords[0]],
        }
    }
}

impl<T> std::ops::Mul<[T; 3]> for &MixType
where
    T: Sized + Copy + std::ops::Neg<Output = T>,
{
    type Output = [T; 3];

    fn mul(self, rhs: [T; 3]) -> Self::Output { self.apply(rhs) }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Transform {
    mix: MixType,
    invert_sign: [bool; 3],
}

impl Transform {
    pub fn identity() -> Transform {
        Self {
            mix: MixType::XYZ,
            invert_sign: [false, false, false],
        }
    }

    pub fn from_byte(v: u8) -> Option<Transform> {
        let invert_sign = [
            ((v >> 4) & 1) == 1, //
            ((v >> 5) & 1) == 1,
            ((v >> 6) & 1) == 1,
        ];

        match v & 0xF {
            0b01_00 => Some(Self { mix: MixType::XYZ, invert_sign }),
            0b00_01 => Some(Self { mix: MixType::YXZ, invert_sign }),
            0b10_00 => Some(Self { mix: MixType::XZY, invert_sign }),
            0b10_01 => Some(Self { mix: MixType::YZX, invert_sign }),
            0b00_10 => Some(Self { mix: MixType::ZXY, invert_sign }),
            0b01_10 => Some(Self { mix: MixType::ZYX, invert_sign }),
            _ => None,
        }
    }

    pub fn apply<T: Sized + Copy + std::ops::Neg<Output = T>>(&self, coords: [T; 3]) -> [T; 3] {
        let mixed = self.mix.apply(coords);

        [
            if self.invert_sign[0] { -mixed[0] } else { mixed[0] },
            if self.invert_sign[1] { -mixed[1] } else { mixed[1] },
            if self.invert_sign[2] { -mixed[2] } else { mixed[2] },
        ]
    }
}

impl<T> std::ops::Mul<[T; 3]> for &Transform
where
    T: Sized + Copy + std::ops::Neg<Output = T>,
{
    type Output = [T; 3];

    fn mul(self, rhs: [T; 3]) -> Self::Output { self.apply(rhs) }
}
