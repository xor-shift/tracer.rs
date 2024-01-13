// #![feature(adt_const_params)]
#![feature(auto_traits)]
#![feature(const_trait_impl)]
#![feature(generic_const_exprs)]
#![feature(let_chains)]
#![feature(macro_metavar_expr)]
#![feature(negative_impls)]
#![feature(stmt_expr_attributes)]
#![allow(incomplete_features)]
#![allow(mixed_script_confusables)]

use std::sync::Arc;
use std::sync::Mutex;

mod camera;
mod color;
mod intersection;
mod material;
mod ray;
mod renderer;
mod scene;
mod shape;
mod types;

pub use camera::*;
pub use color::*;
pub use intersection::*;
pub use material::*;
pub use ray::*;
pub use renderer::*;
pub use scene::*;
pub use shape::*;
pub use types::*;

mod fun;

use stuff::rng::distributions::sphere::NDSampler;
use stuff::rng::distributions::GenerateCanonical;
use stuff::rng::{RandomNumberEngine, UniformRandomBitGenerator};

const MATERIALS: [EMaterial; 6] = [
    EMaterial::LambertianDiffuseIS(material::LambertianIS {
        albedo: LinearRGB(Vec3::new([0., 0., 0.])),
        emittance: LinearRGB(Vec3::new_explode(12.)),
    }),
    EMaterial::PerfectMirror(material::PerfectMirror{}),
    EMaterial::PerfectDielectric(material::PerfectDielectric{index_of_refraction: 1.7}),
    EMaterial::LambertianDiffuseIS(material::LambertianIS {
        albedo: LinearRGB(Vec3::new_explode(0.75)),
        emittance: LinearRGB(Vec3::new_explode(0.)),
    }),
    EMaterial::LambertianDiffuseIS(material::LambertianIS {
        albedo: LinearRGB(Vec3::new([0.25, 0.25, 0.75])),
        emittance: LinearRGB(Vec3::new_explode(0.)),
    }),
    EMaterial::LambertianDiffuseIS(material::LambertianIS {
        albedo: LinearRGB(Vec3::new([0.75, 0.25, 0.25])),
        emittance: LinearRGB(Vec3::new_explode(0.)),
    }),
];

#[rustfmt::skip]
const SPHERES: [ShapeWithMaterial<Sphere>; 3] = [
    // light
    ShapeWithMaterial(Sphere {
        center: Vec3::new([0., 42.49, 15.]),
        radius: 40.,
    }, 0),
    // mirror
    ShapeWithMaterial(Sphere {
        center: Vec3::new([-1.75, -2.5 + 0.9, 17.5]),
        radius: 0.9,
    }, 1),
    // glass
    ShapeWithMaterial(Sphere {
        center: Vec3::new([1.75, -2.5 + 0.9 + 0.2, 16.5]),
        radius: 0.9,
    }, 2),
];

#[rustfmt::skip]
const PLANES: [ShapeWithMaterial<Plane>; 5] = [
    ShapeWithMaterial(Plane { // ceiling
        center: Vec3::new([0., 2.5, 0.]),
        normal: REVec3(Vec3::new([0., -1., 0.])),
        uv_scale: 1.,
    }, 3),
    ShapeWithMaterial(Plane { // floor
        center: Vec3::new([0., -2.5, 0.]),
        normal: REVec3(Vec3::new([0., 1., 0.])),
        uv_scale: 1.,
    }, 3),
    ShapeWithMaterial(Plane { // back wall
        center: Vec3::new([0., 0., 20.]),
        normal: REVec3(Vec3::new([0., 0., -1.])),
        uv_scale: 1.,
    }, 3),
    ShapeWithMaterial(Plane { // right wall
        center: Vec3::new([3.5, 0., 0.]),
        normal: REVec3(Vec3::new([-1., 0., 0.])),
        uv_scale: 1.,
    }, 4),
    ShapeWithMaterial(Plane { // left wall
        center: Vec3::new([-3.5, 0., 0.]),
        normal: REVec3(Vec3::new([1., 0., 0.])),
        uv_scale: 1.,
    }, 5),
];

fn trace_iterative<T: Intersectable + 'static, Gen: stuff::rng::UniformRandomBitGenerator>(mut ray: Ray, scene: &T, gen: &mut Gen) -> LinearRGB {
    let mut attenuation = Vec3::new_explode(1.);
    let mut light = Vec3::new_explode(0.);

    for depth in 0.. {
        let intersection = if let Some(intersection) = scene.intersect(&ray, IntersectionRequirements::all(), &None) {
            intersection
        } else {
            break;
        };

        let interaction = MATERIALS[intersection.material_id as usize].interact(&intersection, gen);
        
        ray = Ray::new(intersection.position + interaction.wi.0 * 0.000001, interaction.wi);

        light = light + interaction.emittance.0 * attenuation;

        let cur_attenuation = interaction.attenuation.0 * interaction.weight;
        attenuation = cur_attenuation * attenuation;

        if depth > 4 && attenuation.length() < 0.2 {
            let p = attenuation[0].max(attenuation[1]).max(attenuation[2]);
            if p == 0. {
                break;
            }

            if f64::generate_canonical(gen).get() < p {
                attenuation = attenuation / (1. - p);
            } else {
                break;
            }
        }
    }

    color::LinearRGB(light)
}

fn kernel_iterative<Cam: Camera, Gen: stuff::rng::UniformRandomBitGenerator>(pos: (usize, usize), samples: usize, cam: &Cam, gen: &mut Gen) -> color::LinearRGB {
    let res = (0..samples)
        .map(|_| {
            let (ray, ray_pdf) = cam.sample_pixel(pos, Vec3::new_explode(0.), gen);
            let res = trace_iterative(ray, &(&PLANES, &SPHERES), gen).0 / ray_pdf;
            res
        })
        .fold(Vec3::new_explode(0.), std::ops::Add::add)
        / samples as Float;

    let res = color::LinearRGB(res);

    res
}

pub fn main() {
    color_backtrace::install();

    let threads = std::thread::available_parallelism().unwrap().get();
    // let threads = 1;
    let dims = (768 / 2, 512 / 2);
    let samples = 256;
    let mut image = stuff::qoi::Image::new(dims.0, dims.1);

    let mut rd = stuff::rng::engines::RandomDevice::new();
    let mut gen = stuff::rng::engines::Xoshiro256P::new();
    gen.seed_from_result(rd.generate());

    let cam = camera::PinholeCamera::new((image.width() as usize, image.height() as usize), 45_f64.to_radians());

    std::thread::scope(|scope| {
        struct Product {
            for_row: usize,
            pixels: Vec<color::LinearRGB>,
        }
        let row = Arc::new(Mutex::new(0));
        let (sender, receiver) = std::sync::mpsc::channel();
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let row = row.clone();
                let sender = sender.clone();

                gen.discard(192);
                let mut gen = gen.clone();

                let cam = &cam;

                scope.spawn(move || loop {
                    let row = {
                        let mut row = row.lock().unwrap();
                        *row += 1;
                        *row - 1
                    };

                    if row >= dims.1 {
                        break;
                    }

                    let pixels = (0..dims.0) //
                        .map(|col| kernel_iterative((col as usize, row as usize), samples, cam, &mut gen))
                        .collect();

                    sender.send(Product { for_row: row as usize, pixels }).unwrap();
                })
            })
            .collect();

        drop(sender);

        while let Ok(product) = receiver.recv() {
            println!("processing row {}", product.for_row);

            for (x, res) in product.pixels.into_iter().enumerate() {
                let res = color::SRGB::from(res);
                let res = stuff::qoi::Color {
                    r: (res.0[0] * 255.).clamp(0., 255.).round() as u8,
                    g: (res.0[1] * 255.).clamp(0., 255.).round() as u8,
                    b: (res.0[2] * 255.).clamp(0., 255.).round() as u8,
                    a: 255,
                };
                *image.pixel_mut(x, product.for_row) = res;
            }
        }

        workers.into_iter().for_each(|v| v.join().unwrap());
    });

    /*for row in 0..image.height() {
        println!("processing row {}", row);
        for col in 0..image.width() {
            let res = kernel_iterative((col as usize, row as usize), (image.width() as usize, image.height() as usize), samples, &cam, &mut gen);

            let res = color::SRGB::from(res);

            let res = stuff::qoi::Color {
                r: (res.0[0] * 255.).clamp(0., 255.).round() as u8,
                g: (res.0[1] * 255.).clamp(0., 255.).round() as u8,
                b: (res.0[2] * 255.).clamp(0., 255.).round() as u8,
                a: 255,
            };

            *image.pixel_mut(col as usize, row as usize) = res;
        }
    }*/

    let mut file = std::fs::File::create("out.qoi").unwrap();
    image.encode_to_writer(&mut file).unwrap();
}
