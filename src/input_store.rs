use winit::event::ElementState;
use winit::keyboard::Key;

#[derive(Clone, Copy)]
struct FullKeyState {
    state: ElementState,
    last_change: std::time::Instant,
}

impl FullKeyState {
    fn new() -> FullKeyState {
        Self {
            state: ElementState::Released,
            last_change: std::time::Instant::now(),
        }
    }

    fn update(&mut self, state: ElementState) {
        if state == self.state {
            return;
        }

        self.state = state;
        self.last_change = std::time::Instant::now();
    }

    fn pressed_for(&self) -> Option<f64> {
        if self.state == ElementState::Released {
            None
        } else {
            Some((std::time::Instant::now() - self.last_change).as_secs_f64())
        }
    }
}

pub struct InputStore {
    states: std::collections::BTreeMap<winit::keyboard::Key, FullKeyState>,
}

impl InputStore {
    pub fn new() -> InputStore { Self { states: std::collections::BTreeMap::new() } }

    pub fn process_event(&mut self, event: &winit::event::WindowEvent) {
        let (_device_id, key_event, _is_synthetic) = if let winit::event::WindowEvent::KeyboardInput { device_id, event, is_synthetic } = event {
            (device_id, event, is_synthetic)
        } else {
            return;
        };

        self.states //
            .entry(key_event.logical_key.clone())
            .or_insert(FullKeyState::new())
            .update(key_event.state);
    }

    pub fn is_pressed(&self, key: winit::keyboard::Key) -> bool { self.states.get(&key).map(|state| state.state == ElementState::Pressed ).unwrap_or(false) }

    pub fn pressed_for(&self, key: winit::keyboard::Key) -> Option<f64> { self.states.get(&key).and_then(|state: &FullKeyState| state.pressed_for() ) }
}
