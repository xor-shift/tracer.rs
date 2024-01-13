use stuff::rng::distributions::GenerateCanonical;

use crate::*;

pub trait Camera {
    fn sample_pixel<Gen: stuff::rng::UniformRandomBitGenerator>(&self, screen_coords: (usize, usize), pos: Vec3, gen: &mut Gen) -> (Ray, Float);
}
#[derive(Clone)]

pub struct PinholeCamera {
    screen_dims: (usize, usize),
    fov: Float,
}

impl PinholeCamera {
    pub const fn new(screen_dims: (usize, usize), fov: Float) -> Self { Self { screen_dims, fov } }
}

impl Camera for PinholeCamera {
    fn sample_pixel<Gen: stuff::rng::UniformRandomBitGenerator>(&self, screen_coords: (usize, usize), pos: Vec3, gen: &mut Gen) -> (Ray, Float) {
        let half_θ = self.fov / 2.;
        //let d = screen_coords.0 as f64 * (half_θ.atan()) / 2f64 ;
        let d = (1. / (2. * half_θ.sin())) * ((self.screen_dims.0 as Float * (2. - self.screen_dims.0 as Float)).abs()).sqrt();

        let offset = (Float::generate_canonical(gen) * 2. - 1., Float::generate_canonical(gen) * 2. - 1.);

        let direction = Vec3::new([
            screen_coords.0 as Float - (self.screen_dims.0 as Float / 2.) + offset.0, //
            (self.screen_dims.1 - screen_coords.1 - 1) as Float - (self.screen_dims.1 as Float / 2.) as Float + offset.1,
            d,
        ])
        .normalized();

        (Ray::new(pos, REVec3(direction)), 1.)
    }
}
