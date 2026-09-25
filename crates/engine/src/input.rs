//! Per-frame input state, fed with already-decoded events (winit's event
//! structs have private fields, so the app decodes them and calls these methods).

use std::collections::{HashMap, HashSet};

use glam::Vec2;
use winit::event::{MouseButton, TouchPhase};
use winit::keyboard::KeyCode;

#[derive(Debug, Default)]
pub struct InputState {
    keys_held: HashSet<KeyCode>,
    keys_pressed: HashSet<KeyCode>,
    buttons_held: HashSet<MouseButton>,
    cursor: Option<Vec2>,
    cursor_delta: Vec2,
    raw_delta: Vec2,
    wheel: f32,
    touches: HashMap<u64, Vec2>,
    touch_rotate: Vec2,
    touch_pinch: f32,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            touch_pinch: 1.0,
            ..Self::default()
        }
    }

    pub fn on_key(&mut self, key: KeyCode, pressed: bool) {
        if pressed {
            if self.keys_held.insert(key) {
                self.keys_pressed.insert(key);
            }
        } else {
            self.keys_held.remove(&key);
        }
    }

    pub fn on_mouse_button(&mut self, button: MouseButton, pressed: bool) {
        if pressed {
            self.buttons_held.insert(button);
        } else {
            self.buttons_held.remove(&button);
        }
    }

    pub fn on_cursor_moved(&mut self, position: Vec2) {
        if let Some(previous) = self.cursor {
            self.cursor_delta += position - previous;
        }
        self.cursor = Some(position);
    }

    pub fn on_cursor_left(&mut self) {
        self.cursor = None;
    }

    /// Positive = scroll up / away from the user.
    pub fn on_wheel(&mut self, lines: f32) {
        self.wheel += lines;
    }

    /// Unaccelerated mouse motion (works while the pointer is locked).
    pub fn on_raw_motion(&mut self, delta: Vec2) {
        self.raw_delta += delta;
    }

    pub fn on_touch(&mut self, id: u64, phase: TouchPhase, position: Vec2) {
        match phase {
            TouchPhase::Started => {
                self.touches.insert(id, position);
            }
            TouchPhase::Moved => {
                let before = self.pinch_distance();
                let Some(old) = self.touches.insert(id, position) else {
                    return;
                };
                match self.touches.len() {
                    1 => self.touch_rotate += position - old,
                    2 => {
                        if let (Some(before), Some(after)) = (before, self.pinch_distance())
                            && before > 0.0
                        {
                            self.touch_pinch *= after / before;
                        }
                    }
                    _ => {}
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.touches.remove(&id);
            }
        }
    }

    fn pinch_distance(&self) -> Option<f32> {
        let mut points = self.touches.values();
        match (points.next(), points.next(), points.next()) {
            (Some(a), Some(b), None) => Some(a.distance(*b)),
            _ => None,
        }
    }

    pub fn key_held(&self, key: KeyCode) -> bool {
        self.keys_held.contains(&key)
    }

    /// True only on the frame the key went down (ignores OS key repeat).
    pub fn key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    pub fn button_held(&self, button: MouseButton) -> bool {
        self.buttons_held.contains(&button)
    }

    pub fn cursor_delta(&self) -> Vec2 {
        self.cursor_delta
    }

    pub fn raw_delta(&self) -> Vec2 {
        self.raw_delta
    }

    pub fn wheel(&self) -> f32 {
        self.wheel
    }

    /// One-finger drag this frame, in pixels.
    pub fn touch_rotate(&self) -> Vec2 {
        self.touch_rotate
    }

    /// Two-finger spread this frame as a ratio (> 1 = fingers apart).
    pub fn touch_pinch(&self) -> f32 {
        self.touch_pinch
    }

    /// Releases everything held, e.g. when the window loses focus.
    pub fn release_all(&mut self) {
        self.keys_held.clear();
        self.buttons_held.clear();
        self.touches.clear();
    }

    pub fn end_frame(&mut self) {
        self.keys_pressed.clear();
        self.cursor_delta = Vec2::ZERO;
        self.raw_delta = Vec2::ZERO;
        self.wheel = 0.0;
        self.touch_rotate = Vec2::ZERO;
        self.touch_pinch = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_track_held_and_single_frame_presses() {
        let mut input = InputState::new();
        input.on_key(KeyCode::KeyW, true);
        assert!(input.key_held(KeyCode::KeyW) && input.key_pressed(KeyCode::KeyW));
        input.end_frame();
        input.on_key(KeyCode::KeyW, true); // OS key repeat
        assert!(input.key_held(KeyCode::KeyW) && !input.key_pressed(KeyCode::KeyW));
        input.on_key(KeyCode::KeyW, false);
        assert!(!input.key_held(KeyCode::KeyW));
    }

    #[test]
    fn cursor_deltas_accumulate_and_reset() {
        let mut input = InputState::new();
        input.on_cursor_moved(Vec2::new(10.0, 10.0));
        assert_eq!(input.cursor_delta(), Vec2::ZERO, "first move has no delta");
        input.on_cursor_moved(Vec2::new(13.0, 8.0));
        input.on_cursor_moved(Vec2::new(15.0, 8.0));
        assert_eq!(input.cursor_delta(), Vec2::new(5.0, -2.0));
        input.on_wheel(1.5);
        input.on_raw_motion(Vec2::new(4.0, 1.0));
        input.end_frame();
        assert_eq!(input.cursor_delta(), Vec2::ZERO);
        assert_eq!(input.wheel(), 0.0);
        assert_eq!(input.raw_delta(), Vec2::ZERO);
    }

    #[test]
    fn one_finger_rotates_two_fingers_pinch() {
        let mut input = InputState::new();
        input.on_touch(1, TouchPhase::Started, Vec2::new(0.0, 0.0));
        input.on_touch(1, TouchPhase::Moved, Vec2::new(5.0, 2.0));
        assert_eq!(input.touch_rotate(), Vec2::new(5.0, 2.0));

        input.on_touch(2, TouchPhase::Started, Vec2::new(105.0, 2.0)); // 100 px apart
        input.on_touch(2, TouchPhase::Moved, Vec2::new(205.0, 2.0)); // 200 px apart
        assert!((input.touch_pinch() - 2.0).abs() < 1e-5);
        assert_eq!(
            input.touch_rotate(),
            Vec2::new(5.0, 2.0),
            "pinch must not rotate"
        );

        input.end_frame();
        input.on_touch(2, TouchPhase::Ended, Vec2::new(205.0, 2.0));
        input.on_touch(1, TouchPhase::Moved, Vec2::new(8.0, 2.0));
        assert_eq!(input.touch_pinch(), 1.0);
        assert_eq!(input.touch_rotate(), Vec2::new(3.0, 0.0));
    }

    #[test]
    fn release_all_clears_held_state() {
        let mut input = InputState::new();
        input.on_key(KeyCode::KeyW, true);
        input.on_mouse_button(MouseButton::Left, true);
        input.release_all();
        assert!(!input.key_held(KeyCode::KeyW));
        assert!(!input.button_held(MouseButton::Left));
    }
}
