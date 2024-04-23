#![allow(incomplete_features)]
#![feature(const_mut_refs)]
#![feature(generic_const_exprs)]
#![feature(iter_partition_in_place)]
#![feature(num_midpoint)]

pub mod basic_octree;
pub mod shapes;

pub struct Ray<S> {
    origin: cgmath::Point3<S>,
    direction: cgmath::Vector3<S>,
    direction_reciprocals: cgmath::Vector3<S>,
}

impl<S: Copy + cgmath::One + std::ops::Div<Output = S>> Ray<S> {
    pub fn new(origin: cgmath::Point3<S>, direction: cgmath::Vector3<S>) -> Ray<S> {
        Self {
            origin,
            direction,
            direction_reciprocals: direction.map(|v| S::one() / v),
        }
    }
}

pub struct BasicIntersection<S> {
    pub global_point: cgmath::Point3<S>,
    pub distance: S,

    pub wo: cgmath::Vector3<S>,
    pub normal: cgmath::Vector3<S>,
}

pub trait HitCheckable {
    fn hit_check<S>(&self, ray: &Ray<S>) -> bool;
}

pub trait BasicIntersectable: HitCheckable {
    fn basic_intersect<S>(&self, ray: &Ray<S>) -> BasicIntersection<S>;
}

pub trait Bounded<S> {
    fn get_bounds(&self) -> shapes::AABBox<S>;
}
