use winit::event::ElementState;
use winit::keyboard::Key;

#[derive(Clone, Copy)]
struct KeyState {
    state: ElementState,
    last_change: std::time::Instant,
}

impl KeyState {
    fn new() -> KeyState {
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

#[derive(Clone)]
struct MouseState {
    pressed_buttons: std::collections::BTreeSet<winit::event::MouseButton>,

    last_location: winit::dpi::PhysicalPosition<f64>,
    curr_location: winit::dpi::PhysicalPosition<f64>,

    drag_start: Option<std::time::Instant>,
}

impl MouseState {
    pub fn new() -> MouseState {
        Self {
            pressed_buttons: std::default::Default::default(),

            last_location: (0., 0.).into(),
            curr_location: (0., 0.).into(),

            drag_start: None,
        }
    }

    fn frame_start(&mut self) {}

    fn update_click(&mut self, state: winit::event::ElementState, button: winit::event::MouseButton) {
        if state.is_pressed() {
            self.pressed_buttons.insert(button);

            self.drag_start = Some(self.drag_start.unwrap_or(std::time::Instant::now()));
        } else {
            self.pressed_buttons.remove(&button);

            if self.pressed_buttons.is_empty() {
                self.drag_start = None;
            }
        }
    }

    fn update_move(&mut self, position: winit::dpi::PhysicalPosition<f64>) { self.curr_location = position; }

    fn drag_is_starting(&self) -> bool { self.drag_start.is_none() && !self.pressed_buttons.is_empty() }
    fn drag_is_ending(&self) -> bool { self.drag_start.is_some() && self.pressed_buttons.is_empty() }

    fn frame_end(&mut self) { self.last_location = self.curr_location; }
}

pub struct InputStore {
    states: std::collections::BTreeMap<winit::keyboard::Key, KeyState>,

    mouse_state: MouseState,
}

impl InputStore {
    pub fn new() -> InputStore {
        Self {
            states: std::collections::BTreeMap::new(),
            mouse_state: MouseState::new(),
        }
    }

    pub fn frame_start(&mut self) {
        self.mouse_state.frame_start(); //
    }

    pub fn process_event(&mut self, event: &winit::event::WindowEvent) {
        match event {
            winit::event::WindowEvent::KeyboardInput { device_id: _, event, is_synthetic: _ } => {
                self.states //
                    .entry(event.logical_key.clone())
                    .or_insert(KeyState::new())
                    .update(event.state);
            }

            winit::event::WindowEvent::MouseInput { device_id: _, state, button } => {
                self.mouse_state.update_click(*state, *button);
            }

            winit::event::WindowEvent::CursorMoved { device_id: _, position } => {
                self.mouse_state.update_move(*position);
            }

            _ => {}
        }
    }

    pub fn frame_end(&mut self) {
        self.mouse_state.frame_end(); //
    }

    pub fn mouse_location(&self) -> (f64, f64) { self.mouse_state.curr_location.into() }

    pub fn mouse_move(&self) -> (f64, f64) { (self.mouse_state.curr_location.x - self.mouse_state.last_location.x, self.mouse_state.curr_location.y - self.mouse_state.last_location.y) }

    pub fn mouse_move_drag(&self) -> (f64, f64) {
        if self.mouse_state.drag_start.is_some() {
            (self.mouse_state.curr_location.x - self.mouse_state.last_location.x, self.mouse_state.curr_location.y - self.mouse_state.last_location.y)
        } else {
            (0., 0.)
        }
    }

    pub fn is_pressed(&self, key: &winit::keyboard::Key) -> bool { self.states.get(key).map(|state| state.state == ElementState::Pressed).unwrap_or(false) }

    pub fn pressed_for(&self, key: &winit::keyboard::Key) -> Option<f64> { self.states.get(key).and_then(|state: &KeyState| state.pressed_for()) }
}
