pub use contracts::debug_ensures;
pub use contracts::debug_invariant;
pub use contracts::debug_requires;
pub use contracts::ensures;
pub use contracts::invariant;
pub use contracts::requires;

pub type Float = f64;
pub type Vec2 = stuff::smallvec::Vector<Float, 2>;
pub type Vec3 = stuff::smallvec::Vector<Float, 3>;
pub type Vec4 = stuff::smallvec::Vector<Float, 4>;

pub type Mat3x3 = stuff::smallvec::Matrix<Float, 3, 3>;
pub type Mat4x4 = stuff::smallvec::Matrix<Float, 4, 4>;

pub fn orthonormal_from_xz(x: Vec3, z: Vec3) -> Mat3x3 {
    let y = z.cross(x);

    #[rustfmt::skip]
    Mat3x3::new([
        x[0], x[1], x[2],
        y[0], y[1], y[2],
        z[0], z[1], z[2],
    ])
}

/// spaces:
/// name  normalised  "up"       dims  description
///       no          global y   any   global space, default
/// SH    no          global y   3, 4  shape space
/// RE    yes         global y   3, 4  reflection space
/// S     yes         {0, 0, 1}  3, 4  surface space (normal is always {0, 0, 1})
/// RN    depends     {0, 0, 1}  3     rng space
/// UV    no          n/a        2     texture space (2D only)
///
/// a "up" of a space means that the "normal" of the plane that arbitrarily
/// defines the "surface" also defines the "up" direction.
///
/// the global "up" is +y, or {0, 1, 0}
///
/// the normality of a vector in RN depends on the context. sphere
/// generators should return normal vectors, as opposed to ball generators
///

#[inline]
pub fn is_normalised(vec: Vec3) -> bool {
    let len = vec.length();
    let error = (len - 1.).abs();
    error < 0.001
}

#[inline]
pub fn vectors_are_same(lhs: Vec3, rhs: Vec3) -> bool { true }

#[derive(Clone, Copy, PartialEq)]
pub struct VecUV(pub Vec2);

impl VecUV {
    #[inline]
    pub fn is_ok(&self) -> bool { self.0[0] >= 0. && self.0[0] <= 1. && self.0[1] >= 0. && self.0[1] <= 1. }
}

#[derive(Clone, Copy, PartialEq)]
pub struct SHVec3(pub Vec3);

#[derive(Clone, Copy, PartialEq)]
pub struct SHVec4(pub Vec4);

#[derive(Clone, Copy, PartialEq)]
pub struct REVec3(pub Vec3);

impl REVec3 {
    #[inline]
    pub fn is_ok(&self) -> bool { is_normalised(self.0) }
}

#[derive(Clone, Copy, PartialEq)]
pub struct REVec4(pub Vec4);

#[derive(Clone, Copy, PartialEq)]
pub struct SVec3(pub Vec3);

#[derive(Clone, Copy, PartialEq)]
pub struct SVec4(pub Vec4);

#[derive(Clone, Copy, PartialEq)]
pub struct RNVec2(pub Vec2);

#[derive(Clone, Copy, PartialEq)]
pub struct RNVec3(pub Vec3);

#[derive(Clone, Copy, PartialEq)]
pub struct RNVec4(pub Vec4);

#[derive(Clone, Copy, PartialEq)]
pub struct Extent3D(pub Vec3, pub Vec3);

#[debug_invariant(self.0[0] <= self.1[0])]
#[debug_invariant(self.0[1] <= self.1[1])]
#[debug_invariant(self.0[2] <= self.1[2])]
impl Extent3D {
    pub fn new_infinite() -> Self {
        Self(
            Vec3::new_explode(Float::INFINITY), //
            Vec3::new_explode(Float::INFINITY),
        )
    }

    pub fn extend(self, other: Extent3D) -> Extent3D {
        Self(
            Vec3::new([
                self.0[0].min(other.0[0]), //
                self.0[1].min(other.0[1]),
                self.0[2].min(other.0[2]),
            ]),
            Vec3::new([
                self.1[0].max(other.1[0]), //
                self.1[1].max(other.1[1]),
                self.1[2].max(other.1[2]),
            ]),
        )
    }

    pub fn center(&self) -> Vec3 { (self.1 + self.0) / 2. }
}

impl std::default::Default for Extent3D {
    fn default() -> Self { Extent3D(Vec3::new_explode(0.), Vec3::new_explode(0.)) }
}
