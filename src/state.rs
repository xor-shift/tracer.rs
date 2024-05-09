use cgmath::SquareMatrix;
use stuff::rng::{RandomNumberEngine, UniformRandomBitGenerator};

use crate::input_tracker;

#[derive(Clone, Copy)]
pub enum VisualisationMode {
    PathTrace,
    RealIntersectionTests,
    BoundIntersectionTests,
    Denoise0,
    Denoise1,
    PathTraceAlbedo,
    Denoise0Albedo,
    Denoise1Albedo,
    Normal,
    AbsNormal,
    DistFromOrigin,
}

impl VisualisationMode {
    pub const fn get_arr() -> [(VisualisationMode, &'static str); 11] {
        use VisualisationMode::*;
        [
            (PathTrace, "pt output"),
            (RealIntersectionTests, "real intersection tests"),
            (BoundIntersectionTests, "bound intersection tests"),
            (Denoise0, "denoise buffer 0"),
            (Denoise1, "denoise buffer 1"),
            (PathTraceAlbedo, "pt output + albedo"),
            (Denoise0Albedo, "denoise buffer 0 + albedo"),
            (Denoise1Albedo, "denoise buffer 1 + albedo"),
            (Normal, "surface normals"),
            (AbsNormal, "absolute surface normals"),
            (DistFromOrigin, "distance from origin"),
        ]
    }

    pub fn as_str(&self) -> &'static str { Self::get_arr()[*self as usize].1 }

    pub fn prev(&self) -> VisualisationMode {
        let arr = Self::get_arr();
        arr[(*self as usize + arr.len() - 1) % arr.len()].0
    }
    pub fn next(&self) -> VisualisationMode {
        let arr = Self::get_arr();
        arr[(*self as usize + 1) % arr.len()].0
    }
}

pub struct RealtimePTConfig {}

#[derive(Clone, Copy)]
pub enum Renderer {
    Raster,
    AlmostRaster,
    RealtimePT,
    PathTracer,
}

pub struct State {
    pub visualisation_mode: VisualisationMode,
    pub position: cgmath::Point3<f64>,
    pub rotation: cgmath::Vector3<f64>,

    pub fov: cgmath::Deg<f64>,

    pub mouse_accel_enabled: bool,
    pub mouse_accel_param: f64,
    pub linear_mouse_speed: f64,

    last_frame: std::time::Instant,
    started_at: std::time::Instant,
    frame_no: u32,
    frame_seed_generator: stuff::rng::engines::Xoshiro256PP,

    dimensions: (u32, u32),
}

impl State {
    pub fn new() -> State {
        let mut frame_seed_generator = stuff::rng::engines::Xoshiro256PP::new();
        let mut random_device = stuff::rng::engines::RandomDevice::new();
        frame_seed_generator.seed_from_result(random_device.generate());
        drop(random_device);

        Self {
            visualisation_mode: VisualisationMode::PathTrace,
            position: cgmath::point3(0., 64., 0.),
            rotation: cgmath::vec3(0., 0., 0.),

            fov: cgmath::Deg(30.),

            mouse_accel_enabled: false,
            mouse_accel_param: 4.,
            linear_mouse_speed: 10.,

            last_frame: std::time::Instant::now(),
            started_at: std::time::Instant::now(),
            frame_no: 0,
            frame_seed_generator,

            dimensions: (0, 0),
        }
    }

    pub fn frame_start(&mut self) {}

    pub fn window_event(&mut self, event: &winit::event::WindowEvent) {
        match event {
            winit::event::WindowEvent::KeyboardInput { device_id: _, event, is_synthetic: _ } if event.state.is_pressed() => {
                if event.logical_key == "q" {
                    self.visualisation_mode = self.visualisation_mode.prev();
                } else if event.logical_key == "e" {
                    self.visualisation_mode = self.visualisation_mode.next();
                }
            }

            winit::event::WindowEvent::Resized(new_size) => {
                self.dimensions = (*new_size).into();
            }

            _ => {}
        }
    }

    fn process_rotation(&mut self, input: &input_tracker::InputTracker, delta_t: std::time::Duration) -> cgmath::Matrix3<f64> {
        let delta_rot = if input.mouse_is_pressed(winit::event::MouseButton::Left) { input.mouse_velocity() } else { cgmath::vec2(0., 0.) };
        let delta_rot = if self.mouse_accel_enabled {
            delta_rot.map(|v| {
                let max_delta = 64.;

                let sig = v.signum();
                let v = (v.abs() / max_delta).clamp(0., 1.);

                let res = (v * v * (3. - 2. * v)) * 4.;

                res * max_delta * sig
            })
        } else {
            delta_rot * self.linear_mouse_speed
        };

        self.rotation += cgmath::vec3(delta_rot.x, delta_rot.y, 0.) * delta_t.as_secs_f64();

        #[rustfmt::skip]
        let mat =
            cgmath::Matrix3::from_angle_z(cgmath::Deg(self.rotation[2])) *
            cgmath::Matrix3::from_angle_y(cgmath::Deg(self.rotation[0])) *
            cgmath::Matrix3::from_angle_x(cgmath::Deg(self.rotation[1]));

        mat
    }

    fn process_perspective(&mut self, input: &input_tracker::InputTracker, delta_t: std::time::Duration, rotation: &cgmath::Matrix3<f64>) -> cgmath::Matrix4<f64> {
        let actions: [(winit::keyboard::Key<winit::keyboard::SmolStr>, cgmath::Vector3<f64>); 6] = [
            (winit::keyboard::Key::Character("w".into()), cgmath::vec3(0., 0., 1.)),
            (winit::keyboard::Key::Character("a".into()), cgmath::vec3(-1., 0., 0.)),
            (winit::keyboard::Key::Character("s".into()), cgmath::vec3(0., 0., -1.)),
            (winit::keyboard::Key::Character("d".into()), cgmath::vec3(1., 0., 0.)),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space), cgmath::vec3(0., 1., 0.)),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Shift), cgmath::vec3(0., -1., 0.)),
        ];

        let delta_pos = actions //
            .iter()
            .filter(|&item| input.key_is_pressed(&item.0))
            .map(|item| item.1)
            .reduce(cgmath::ElementWise::add_element_wise)
            .unwrap_or(cgmath::vec3(0., 0., 0.));
        let delta_pos = rotation * (delta_pos * delta_t.as_secs_f64() * 15.);
        self.position += delta_pos;

        #[rustfmt::skip]
        const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f64> = cgmath::Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.5,
            0.0, 0.0, 0.0, 1.0,
        );

        #[rustfmt::skip]
        const HANDEDNESS_SWAP: cgmath::Matrix4<f64> = cgmath::Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        let look_at = cgmath::vec3(0., 0., 1.);
        let look_at = rotation * look_at;
        let look_at = self.position + look_at;

        let view = cgmath::Matrix4::look_at_lh(self.position, look_at, cgmath::vec3(0., 1., 0.));
        let proj = cgmath::perspective(self.fov, self.dimensions.0 as f64 / self.dimensions.1 as f64, 0.01, 1000.);

        OPENGL_TO_WGPU_MATRIX * proj * HANDEDNESS_SWAP * view
    }

    pub fn pre_render(&mut self, input: &input_tracker::InputTracker) -> RawState {
        let now = std::time::Instant::now();
        let delta_t = now - self.last_frame;
        self.last_frame = now;

        let rotation = self.process_rotation(input, delta_t);
        let perspective = self.process_perspective(input, delta_t, &rotation);

        let frame_seed = {
            let arr = [self.frame_seed_generator.generate(), self.frame_seed_generator.generate()];

            [arr[0] as u32, (arr[0] >> 32) as u32, arr[1] as u32, (arr[1] >> 32) as u32]
        };

        RawState {
            camera_transform: perspective.cast().unwrap().into(),
            inverse_transform: perspective.invert().unwrap().cast().unwrap().into(),
            frame_seed,
            camera_position: self.position.cast().unwrap().into(),
            frame_no: self.frame_no,
            dimensions: self.dimensions.into(),
            current_instant: (std::time::Instant::now() - self.started_at).as_secs_f32(),
            visualisation_mode: self.visualisation_mode as i32,
        }
    }

    pub fn frame_end(&mut self) {}
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RawState {
    pub camera_transform: [[f32; 4]; 4],
    pub inverse_transform: [[f32; 4]; 4],
    pub frame_seed: [u32; 4],
    pub camera_position: [f32; 3],
    pub frame_no: u32,
    pub dimensions: [u32; 2],
    pub current_instant: f32,
    pub visualisation_mode: i32,
}
