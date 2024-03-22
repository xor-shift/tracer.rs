struct PressState {
    started_at: std::time::Instant,
    mouse_at_start: winit::dpi::PhysicalPosition<f64>,
    modifiers: winit::keyboard::ModifiersState,
}

impl PressState {
    fn new_press(mouse_at_start: winit::dpi::PhysicalPosition<f64>, modifiers: winit::keyboard::ModifiersState) -> PressState {
        Self {
            started_at: std::time::Instant::now(),
            mouse_at_start,
            modifiers,
        }
    }
}

pub struct InputTracker {
    previous_mouse_pos: winit::dpi::PhysicalPosition<f64>,
    mouse_pos: winit::dpi::PhysicalPosition<f64>,
    keyboard_buttons: std::collections::HashMap<winit::keyboard::Key, PressState>,
    mouse_buttons: std::collections::HashMap<winit::event::MouseButton, PressState>,

    modifiers: winit::keyboard::ModifiersState,

    window_size: winit::dpi::PhysicalSize<u32>,
    window_has_focus: bool,

    frame_pending_events: Vec<winit::event::WindowEvent>,

    frame_end_key_remove_queue: Vec<winit::keyboard::Key>,
    frame_end_button_remove_queue: Vec<winit::event::MouseButton>,
}

impl InputTracker {
    pub fn new() -> Self {
        Self {
            previous_mouse_pos: (0., 0.).into(),
            mouse_pos: (0., 0.).into(),
            keyboard_buttons: std::collections::HashMap::new(),
            mouse_buttons: std::collections::HashMap::new(),

            modifiers: winit::keyboard::ModifiersState::empty(),

            window_size: (0, 0).into(),
            window_has_focus: false,

            frame_pending_events: Vec::new(),

            frame_end_button_remove_queue: Vec::new(),
            frame_end_key_remove_queue: Vec::new(),
        }
    }

    pub fn frame_start(&mut self) {}

    fn actually_process_window_event(&mut self, event: &winit::event::WindowEvent, ignore_mouse: bool, ignore_kb: bool) {
        use winit::event::WindowEvent::*;

        match event {
            Resized(new_size) => self.window_size = *new_size,
            Focused(focused) => self.window_has_focus = *focused,
            ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            CursorMoved { device_id: _, position } if !ignore_mouse => self.mouse_pos = *position,

            MouseInput { device_id: _, state, button } if !ignore_mouse => {
                if state.is_pressed() {
                    self.mouse_buttons.entry(*button).or_insert(PressState::new_press(self.mouse_pos, self.modifiers));
                } else {
                    self.frame_end_button_remove_queue.push(*button);
                }
            }

            KeyboardInput { device_id: _, event, is_synthetic: _ } if !ignore_kb => {
                if event.state.is_pressed() {
                    self.keyboard_buttons.entry(event.logical_key.clone()).or_insert(PressState::new_press(self.mouse_pos, self.modifiers));
                } else {
                    self.frame_end_key_remove_queue.push(event.logical_key.clone())
                }
            }
            _ => {}
        }
    }

    pub fn window_event(&mut self, event: &winit::event::WindowEvent) { self.frame_pending_events.push(event.clone()); }

    pub fn mouse_is_pressed(&self, button: winit::event::MouseButton) -> bool { self.mouse_buttons.contains_key(&button) }

    pub fn mouse_pos(&self) -> cgmath::Vector2<f64> { cgmath::vec2(self.mouse_pos.x, self.mouse_pos.y) }

    pub fn mouse_velocity(&self) -> cgmath::Vector2<f64> { cgmath::vec2(self.mouse_pos.x - self.previous_mouse_pos.x, self.mouse_pos.y - self.previous_mouse_pos.y) }

    pub fn key_is_pressed(&self, key: &winit::keyboard::Key) -> bool { return self.keyboard_buttons.contains_key(key); }

    pub fn frame_process_events(&mut self, ignore_mouse: bool, ignore_kb: bool) {
        let mut events = Vec::new();
        std::mem::swap(&mut self.frame_pending_events, &mut events);
    
        for event in &events {
            self.actually_process_window_event(event, ignore_mouse, ignore_kb);
        }
    }

    pub fn frame_end(&mut self) {
        self.frame_pending_events.clear();

        self.previous_mouse_pos = self.mouse_pos;

        for button in &self.frame_end_button_remove_queue {
            self.mouse_buttons.remove(button);
        }
        self.frame_end_button_remove_queue.clear();

        for key in &self.frame_end_key_remove_queue {
            self.keyboard_buttons.remove(key);
        }
        self.frame_end_key_remove_queue.clear();
    }
}
