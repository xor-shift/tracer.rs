use crate::Application;
use crate::subscriber::Subscriber;

pub struct FPSTracker {
    last_render: std::time::Instant,

    last_second: std::time::Instant,
    frametime_sum_since_last_second: f64,
    frames_since_last_second: usize,
}

impl FPSTracker {
    pub fn new() -> Self {
        Self {
            last_render: std::time::Instant::now(),

            last_second: std::time::Instant::now(),
            frametime_sum_since_last_second: 0.,
            frames_since_last_second: 0,
        }
    }
}

impl Subscriber for FPSTracker {
    fn render(&mut self, app: &mut Application, view: &wgpu::TextureView, encoder: &mut wgpu::CommandEncoder, delta_time: std::time::Duration) {
        let now = std::time::Instant::now();

        let frametime = (now - self.last_render).as_secs_f64();
        let duration_of_last_second = (now - self.last_second).as_secs_f64();

        self.last_render = now;
        self.frametime_sum_since_last_second += frametime;
        self.frames_since_last_second += 1;

        if duration_of_last_second >= 1. {
            let avg_ft = self.frametime_sum_since_last_second / self.frames_since_last_second as f64;
            log::debug!("{} frames in the last second (ft(avg)={}, fps(ft)={})", self.frames_since_last_second, avg_ft, duration_of_last_second / avg_ft);

            self.last_second = now;
            self.frametime_sum_since_last_second = 0.;
            self.frames_since_last_second = 0;
        }
    }
}
