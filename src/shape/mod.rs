mod disc;
mod plane;
mod sphere;
mod triangle;

use crate::*;
use material::*;
use ray::*;
use types::*;

pub use disc::*;
pub use plane::*;
pub use sphere::*;
pub use triangle::*;

pub struct SurfaceSample {
    pub position: Vec3,

    pub normal: REVec3,
    pub surface_params: [REVec3; 2],

    pub material_id: u16,
    pub texture_coords: VecUV,

    pub probability_of_sample: Float,
}

impl SurfaceSample {
    pub fn generate_interaction<Gen: stuff::rng::UniformRandomBitGenerator>(&self, ray: &Ray, wo: REVec3, materials: &[EMaterial], gen: &mut Gen) -> Interaction {
        todo!() //
    }
}

pub trait Shape: Intersectable {
    fn sample_surface<Gen: stuff::rng::UniformRandomBitGenerator>(&self, gen: &mut Gen) -> SurfaceSample;

    fn surface_area(&self) -> Float;
    fn volume(&self) -> Float;

    fn center(&self) -> Vec3;
    fn global_bounds(&self) -> Extent3D;
}

#[derive(Clone)]
pub struct ShapeWithMaterial<T: Shape>(pub T, pub u16);

impl<T: Shape> Intersectable for ShapeWithMaterial<T> {
    fn intersect(&self, ray: &crate::ray::Ray, reqs: IntersectionRequirements, previous: &Option<Intersection>) -> Option<Intersection> {
        let ret = self.0.intersect(ray, reqs, previous);
        ret.map(|mut v| {
            v.material_id = self.1;
            v
        })
    }
}

impl<T: Shape> Shape for ShapeWithMaterial<T> {
    fn sample_surface<Gen: stuff::rng::UniformRandomBitGenerator>(&self, gen: &mut Gen) -> SurfaceSample {
        let mut ret = self.0.sample_surface(gen);
        ret.material_id = self.1;
        ret
    }

    fn surface_area(&self) -> Float { self.0.surface_area() }
    fn volume(&self) -> Float { self.0.volume() }

    fn center(&self) -> Vec3 { self.0.center() }
    fn global_bounds(&self) -> Extent3D { self.0.global_bounds() }
}

impl<T: Shape> Shape for [T] {
    fn sample_surface<Gen: stuff::rng::UniformRandomBitGenerator>(&self, gen: &mut Gen) -> SurfaceSample { todo!() }

    // it is strongly recommended to cache the result of this
    fn surface_area(&self) -> Float { self.iter().map(|v| v.surface_area()).reduce(std::ops::Add::add).unwrap_or(0.) }

    // it is strongly recommended to cache the result of this
    fn volume(&self) -> Float { self.iter().map(|v| v.volume()).reduce(std::ops::Add::add).unwrap_or(0.) }

    // it is strongly recommended to cache the result of this
    fn center(&self) -> Vec3 { self.global_bounds().center() }

    // it is strongly recommended to cache the result of this
    fn global_bounds(&self) -> Extent3D { self.iter().map(|v| v.global_bounds()).fold(std::default::Default::default(), Extent3D::extend) }
}

impl<T: Shape, const N: usize> Shape for [T; N] {
    fn sample_surface<Gen: stuff::rng::UniformRandomBitGenerator>(&self, gen: &mut Gen) -> SurfaceSample { <[T] as Shape>::sample_surface(self, gen) }

    fn surface_area(&self) -> Float { <[T] as Shape>::surface_area(self) }
    fn volume(&self) -> Float { <[T] as Shape>::volume(self) }

    fn center(&self) -> Vec3 { <[T] as Shape>::center(self) }
    fn global_bounds(&self) -> Extent3D { <[T] as Shape>::global_bounds(self) }
}

macro_rules! generate_shape_tuple {
    (impl $($types:ident) +) => {
        impl<$($types: Shape, )*> Shape for ($($types, )*) {
            fn sample_surface<Gen: stuff::rng::UniformRandomBitGenerator>(&self, gen: &mut Gen) -> SurfaceSample { todo!() }

            fn surface_area(&self) -> Float {
                let mut ret = 0.;
                $(ret += (&self.${index()} as &$types).surface_area();)*
                return ret;
            }

            fn volume(&self) -> Float {
                let mut ret = 0.;
                $(ret += (&self.${index()} as &$types).volume();)*
                return ret;
            }

            fn center(&self) -> Vec3 { self.global_bounds().center() }

            fn global_bounds(&self) -> Extent3D {
                let mut ret = Extent3D::default();
                $(ret = ret.extend((&self.${index()} as &$types).global_bounds());)*
                ret
            }
        }
    };

    () => {};

    ($type:ident $($types:ident) *) => {
        generate_shape_tuple!(impl $type $($types)*);
        generate_shape_tuple!($($types )*);
    };
}

generate_shape_tuple!(A B C D E F G H I J K L M N O P);

impl<T: Shape> Shape for &T {
    fn sample_surface<Gen: stuff::rng::UniformRandomBitGenerator>(&self, gen: &mut Gen) -> SurfaceSample { <T as Shape>::sample_surface(*self, gen) }

    fn surface_area(&self) -> Float { <T as Shape>::surface_area(*self) }
    fn volume(&self) -> Float { <T as Shape>::volume(*self) }

    fn center(&self) -> Vec3 { <T as Shape>::center(*self) }
    fn global_bounds(&self) -> Extent3D { <T as Shape>::global_bounds(*self) }
}

pub(self) fn find_orthogonal(vec: Vec3) -> Vec3 {
    let [a, b, c] = vec.0;

    match (a.abs() < 0.01, b.abs() < 0.01, c.abs() < 0.01) {
        (false, false, false) | (true, _, _) => Vec3::new([0., -c, b]),
        (_, true, _) => Vec3::new([-c, 0., a]),
        (_, _, true) => Vec3::new([-b, a, 0.]),
    }
}

pub(self) fn dummy_dpduv(normal: REVec3) -> (REVec3, REVec3) {
    let ortho = find_orthogonal(normal.0);

    let dpdu = REVec3(ortho.cross(normal.0).normalized());
    let dpdv = REVec3(normal.0.cross(dpdu.0).normalized());

    (dpdu, dpdv)
}
