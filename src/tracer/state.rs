use cgmath::{SquareMatrix, Transform};
use stuff::rng::{RandomNumberEngine, UniformRandomBitGenerator};

use crate::input_tracker::InputTracker;

#[derive(Clone, PartialEq)]
pub(super) struct State {
    // sorted in descending expected frequency of mutation
    started_at: std::time::Instant,
    dimensions: (u32, u32),

    pub visualisation_mode: i32,

    camera_position: cgmath::Point3<f64>,
    camera_rotation: cgmath::Vector3<f64>, // yaw, pitch, roll
    previous_rotation: cgmath::Matrix3<f64>,
    previous_transform: cgmath::Matrix4<f64>,

    last_render_at: std::time::Instant,
    frame_no: u32,
    generator: stuff::rng::engines::Xoshiro256PP,
}

impl State {
    fn generate_rotation(rotation: cgmath::Vector3<f64>) -> cgmath::Matrix3<f64> {
        return // a
            cgmath::Matrix3::from_angle_z(cgmath::Deg(rotation[2])) *
            cgmath::Matrix3::from_angle_y(cgmath::Deg(rotation[0])) *
            cgmath::Matrix3::from_angle_x(cgmath::Deg(rotation[1])) *
            1.;
    }

    fn generate_transform(camera_position: cgmath::Point3<f64>, rotation: cgmath::Matrix3<f64>, dimensions: (u32, u32)) -> cgmath::Matrix4<f64> {
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
        let look_at = camera_position + look_at;

        let view = cgmath::Matrix4::look_at_lh(camera_position, look_at, cgmath::vec3(0., 1., 0.));
        let proj = cgmath::perspective(cgmath::Deg(30.), dimensions.0 as f64 / dimensions.1 as f64, 0.01, 1000.);

        OPENGL_TO_WGPU_MATRIX * proj * HANDEDNESS_SWAP * view
    }

    pub fn new() -> State {
        let mut gen = stuff::rng::engines::Xoshiro256PP::new();
        let mut rd = stuff::rng::engines::RandomDevice::new();
        gen.seed_from_result(rd.generate());

        State {
            started_at: std::time::Instant::now(),
            dimensions: (1, 1),

            visualisation_mode: 0,

            camera_position: cgmath::point3(0., 0., 0.),
            camera_rotation: cgmath::vec3(0., 0., 0.),
            previous_rotation: cgmath::Matrix3::<f64>::identity(),
            previous_transform: Self::generate_transform(cgmath::point3(0., 0., 0.), cgmath::Matrix3::<f64>::identity(), (1, 1)),

            last_render_at: std::time::Instant::now(),
            frame_no: 0,
            generator: gen,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) { self.dimensions = new_size.into(); }

    pub fn frame_start(&mut self, input: &InputTracker) -> RawState {
        let now = std::time::Instant::now();
        let delta_since_last_frame = (now - self.last_render_at).as_secs_f64();
        self.last_render_at = now;

        let actions: &[(winit::keyboard::Key<winit::keyboard::SmolStr>, cgmath::Vector3<f64>, cgmath::Vector3<f64>)] = &[
            (winit::keyboard::Key::Character("w".into()), cgmath::vec3(0., 0., 1.), cgmath::vec3(0., 0., 0.)),
            (winit::keyboard::Key::Character("a".into()), cgmath::vec3(-1., 0., 0.), cgmath::vec3(0., 0., 0.)),
            (winit::keyboard::Key::Character("s".into()), cgmath::vec3(0., 0., -1.), cgmath::vec3(0., 0., 0.)),
            (winit::keyboard::Key::Character("d".into()), cgmath::vec3(1., 0., 0.), cgmath::vec3(0., 0., 0.)),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space), cgmath::vec3(0., 1., 0.), cgmath::vec3(0., 0., 0.)),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Shift), cgmath::vec3(0., -1., 0.), cgmath::vec3(0., 0., 0.)),
            (winit::keyboard::Key::Character("r".into()), cgmath::vec3(0., 0., 0.), cgmath::vec3(0., -1., 0.)),
            (winit::keyboard::Key::Character("f".into()), cgmath::vec3(0., 0., 0.), cgmath::vec3(0., 1., 0.)),
            (winit::keyboard::Key::Character("z".into()), cgmath::vec3(0., 0., 0.), cgmath::vec3(-1., 0., 0.)),
            (winit::keyboard::Key::Character("x".into()), cgmath::vec3(0., 0., 0.), cgmath::vec3(1., 0., 0.)),
        ];

        let (delta_pos, _delta_rot) = actions //
            .iter()
            .filter(|&item| input.key_is_pressed(&item.0))
            .map(|item| (item.1, item.2))
            .reduce(|a, b| (a.0 + b.0, a.1 + b.1))
            .unwrap_or((cgmath::vec3(0., 0., 0.), cgmath::vec3(0., 0., 0.)));

        let delta_rot = if input.mouse_is_pressed(winit::event::MouseButton::Left) { input.mouse_velocity() } else { cgmath::vec2(0., 0.) };

        if delta_pos != cgmath::vec3(0., 0., 0.) || delta_rot != cgmath::vec2(0., 0.) {
            let delta_rot = cgmath::vec3(delta_rot.x as f64, delta_rot.y as f64, 0.);
            self.camera_rotation += delta_rot * delta_since_last_frame * 40.;
            self.previous_rotation = Self::generate_rotation(self.camera_rotation);

            let delta_pos = delta_pos * delta_since_last_frame * 15.;
            let delta_pos = self.previous_rotation * delta_pos;
            self.camera_position += delta_pos;

            let new_transform = Self::generate_transform(self.camera_position, self.previous_rotation, self.dimensions);

            self.previous_transform = new_transform;
        }

        let inverse = self.previous_transform.inverse_transform().unwrap();

        let generated_values = [self.generator.generate(), self.generator.generate()];
        let frame_seed = [(generated_values[0]) as u32, (generated_values[0] >> 32) as u32, (generated_values[1]) as u32, (generated_values[1] >> 32) as u32];

        RawState {
            camera_transform: self.previous_transform.cast::<f32>().unwrap().into(),
            inverse_transform: inverse.cast::<f32>().unwrap().into(),
            frame_seed,
            camera_position: self.camera_position.cast::<f32>().unwrap().into(),
            frame_no: self.frame_no,
            current_instant: (now - self.started_at).as_secs_f32(),
            width: self.dimensions.0,
            height: self.dimensions.1,
            visualisation_mode: self.visualisation_mode,
        }
    }

    pub fn frame_end(&mut self) { self.frame_no += 1; }

    pub fn should_swap_buffers(&self) -> bool { return self.frame_no % 2 == 1; }
}

#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct RawState {
    camera_transform: [[f32; 4]; 4],
    inverse_transform: [[f32; 4]; 4],
    frame_seed: [u32; 4],
    camera_position: [f32; 3],
    frame_no: u32,
    current_instant: f32,
    width: u32,
    height: u32,
    visualisation_mode: i32,
}
