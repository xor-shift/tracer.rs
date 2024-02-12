use super::*;
use crate::*;

bitflags::bitflags! {
    #[derive(Clone, Copy)]
    /// Specifies what information should be gathered by the intersection check.
    ///
    /// Implementors are free to provide excess information.
    ///
    /// Fields:
    /// - empty -> hit check alone
    /// - BASIC -> position, distance, wo (previous hit is checked if BASIC is set)
    /// - NORMAL -> normal
    /// - TEXTURE -> uv coordinates
    /// - SURFACE -> dp_du, dp_dv, surface to reflection space matrix
    /// - STATISTICS -> statistics
    pub struct IntersectionRequirements: u8 {
        const BASIC      = 1 << 0;
        const NORMAL     = 1 << 1;
        const TEXTURE    = 1 << 2;
        const SURFACE    = 1 << 3;
        const STATISTICS = 1 << 4;
    }
}

pub trait Intersectable {
    fn hit_check(&self, ray: &crate::ray::Ray) -> bool { self.intersect(ray, IntersectionRequirements::empty(), &None).is_some() }

    fn intersect(&self, ray: &crate::ray::Ray, reqs: IntersectionRequirements, previous: &Option<Intersection>) -> Option<Intersection>;
}

impl<T: Intersectable> Intersectable for &T {
    fn intersect(&self, ray: &crate::ray::Ray, reqs: IntersectionRequirements, previous: &Option<Intersection>) -> Option<Intersection> {
        <T as Intersectable>::intersect(*self, ray, reqs, previous) //
    }
}

impl<T: Intersectable> Intersectable for [T] {
    fn intersect(&self, ray: &crate::ray::Ray, reqs: IntersectionRequirements, previous: &Option<Intersection>) -> Option<Intersection> {
        let mut res = previous.clone();

        for isectable in self {
            let cur_res = isectable.intersect(ray, reqs, &res);
            res = Intersection::select_best(res, cur_res);
        }

        res
    }
}

impl<T: Intersectable, const N: usize> Intersectable for [T; N] {
    fn intersect(&self, ray: &crate::ray::Ray, reqs: IntersectionRequirements, previous: &Option<Intersection>) -> Option<Intersection> {
        <[T] as Intersectable>::intersect(self, ray, reqs, previous) //
    }
}

macro_rules! generate_intersectable_tuple {
    (impl $($types:ident) +) => {
        impl<$($types: Intersectable, )*> Intersectable for ($($types, )*) {
            fn intersect(&self, ray: &crate::ray::Ray, reqs: IntersectionRequirements, previous: &Option<Intersection>) -> Option<Intersection> {
                let mut res = previous.clone();

                $(
                    let res_tmp = (&self.${index()} as &$types).intersect(ray, reqs, &res);
                    res = Intersection::select_best(res, res_tmp);
                )*

                res
            }
        }
    };

    () => {};

    ($type:ident $($types:ident) *) => {
        generate_intersectable_tuple!(impl $type $($types)*);
        generate_intersectable_tuple!($($types )*);
    };
}

generate_intersectable_tuple!(A B C D E F G H I J K L M N O P);
