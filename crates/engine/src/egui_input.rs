//! Translates window input into `egui::RawInput`.
//!
//! egui-winit (0.36) doesn't build for wasm32, so impulse feeds egui itself.
//! Only what the debug panel needs: pointer, buttons, wheel, focus, and the
//! first touch as a pointer. No keyboard, text, or clipboard.

use egui::{
    Event, Modifiers, MouseWheelUnit, PointerButton, Pos2, RawInput, Rect, TouchPhase, Vec2,
};

#[derive(Debug)]
pub struct EguiInput {
    raw: RawInput,
    pixels_per_point: f32,
    pointer: Option<Pos2>,
    focused: bool,
    touch_id: Option<u64>,
}

impl EguiInput {
    pub fn new(pixels_per_point: f32) -> Self {
        Self {
            raw: RawInput::default(),
            pixels_per_point: sanitize_ppp(pixels_per_point),
            pointer: None,
            focused: true,
            touch_id: None,
        }
    }

    pub fn set_pixels_per_point(&mut self, pixels_per_point: f32) {
        self.pixels_per_point = sanitize_ppp(pixels_per_point);
    }

    fn to_points(&self, physical: (f64, f64)) -> Pos2 {
        Pos2::new(
            physical.0 as f32 / self.pixels_per_point,
            physical.1 as f32 / self.pixels_per_point,
        )
    }

    /// `physical` is the cursor position in physical pixels.
    pub fn on_cursor_moved(&mut self, physical: (f64, f64)) {
        let pos = self.to_points(physical);
        self.pointer = Some(pos);
        self.raw.events.push(Event::PointerMoved(pos));
    }

    pub fn on_cursor_left(&mut self) {
        self.pointer = None;
        self.raw.events.push(Event::PointerGone);
    }

    /// Ignored until the pointer position is known (egui needs a position).
    pub fn on_mouse_button(&mut self, button: PointerButton, pressed: bool) {
        if let Some(pos) = self.pointer {
            self.raw.events.push(Event::PointerButton {
                pos,
                button,
                pressed,
                modifiers: Modifiers::default(),
            });
        }
    }

    pub fn on_wheel_lines(&mut self, x: f32, y: f32) {
        self.push_wheel(MouseWheelUnit::Line, Vec2::new(x, y));
    }

    /// Pixel deltas arrive in physical pixels; egui wants points.
    pub fn on_wheel_pixels(&mut self, x: f64, y: f64) {
        let delta = Vec2::new(x as f32, y as f32) / self.pixels_per_point;
        self.push_wheel(MouseWheelUnit::Point, delta);
    }

    fn push_wheel(&mut self, unit: MouseWheelUnit, delta: Vec2) {
        self.raw.events.push(Event::MouseWheel {
            unit,
            delta,
            phase: TouchPhase::Move,
            modifiers: Modifiers::default(),
        });
    }

    pub fn on_focus(&mut self, focused: bool) {
        self.focused = focused;
        self.raw.events.push(Event::WindowFocused(focused));
    }

    /// The first finger down acts as the mouse, so panel buttons work on phones.
    pub fn on_touch(&mut self, id: u64, phase: TouchPhase, physical: (f64, f64)) {
        match phase {
            TouchPhase::Start if self.touch_id.is_none() => {
                self.touch_id = Some(id);
                self.on_cursor_moved(physical);
                self.on_mouse_button(PointerButton::Primary, true);
            }
            TouchPhase::Move if self.touch_id == Some(id) => self.on_cursor_moved(physical),
            TouchPhase::End | TouchPhase::Cancel if self.touch_id == Some(id) => {
                self.touch_id = None;
                self.on_cursor_moved(physical);
                self.on_mouse_button(PointerButton::Primary, false);
                self.on_cursor_left();
            }
            _ => {}
        }
    }

    /// Everything since the last call, ready for `Context::run_ui`.
    pub fn take(&mut self, size_in_pixels: [u32; 2], time: f64) -> RawInput {
        let mut raw = std::mem::take(&mut self.raw);
        let size = Vec2::new(size_in_pixels[0] as f32, size_in_pixels[1] as f32);
        raw.screen_rect = Some(Rect::from_min_size(
            Pos2::ZERO,
            size / self.pixels_per_point,
        ));
        raw.time = Some(time);
        raw.focused = self.focused;
        raw.viewports
            .entry(raw.viewport_id)
            .or_default()
            .native_pixels_per_point = Some(self.pixels_per_point);
        raw
    }
}

fn sanitize_ppp(pixels_per_point: f32) -> f32 {
    if pixels_per_point.is_finite() && pixels_per_point > 0.0 {
        pixels_per_point
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_is_converted_to_points() {
        let mut input = EguiInput::new(2.0);
        input.on_cursor_moved((200.0, 100.0));
        let raw = input.take([800, 600], 0.0);
        assert_eq!(
            raw.events,
            vec![Event::PointerMoved(Pos2::new(100.0, 50.0))]
        );
    }

    #[test]
    fn button_needs_a_known_pointer_position() {
        let mut input = EguiInput::new(1.0);
        input.on_mouse_button(PointerButton::Primary, true);
        assert!(input.take([10, 10], 0.0).events.is_empty());
        input.on_cursor_moved((3.0, 4.0));
        input.on_mouse_button(PointerButton::Primary, true);
        let events = input.take([10, 10], 0.0).events;
        assert!(matches!(
            events[1],
            Event::PointerButton { pressed: true, .. }
        ));
    }

    #[test]
    fn take_sets_screen_in_points_and_drains_events() {
        let mut input = EguiInput::new(2.0);
        input.on_cursor_moved((1.0, 1.0));
        let raw = input.take([800, 600], 1.5);
        assert_eq!(
            raw.screen_rect,
            Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(400.0, 300.0)))
        );
        assert_eq!(raw.time, Some(1.5));
        assert_eq!(
            raw.viewports[&raw.viewport_id].native_pixels_per_point,
            Some(2.0)
        );
        assert!(input.take([800, 600], 1.6).events.is_empty());
    }

    #[test]
    fn pixel_wheel_is_scaled_to_points() {
        let mut input = EguiInput::new(2.0);
        input.on_wheel_pixels(0.0, 40.0);
        let events = input.take([10, 10], 0.0).events;
        assert!(matches!(
            events[0],
            Event::MouseWheel { unit: MouseWheelUnit::Point, delta, .. } if delta == Vec2::new(0.0, 20.0)
        ));
    }

    #[test]
    fn first_touch_clicks_and_second_is_ignored() {
        let mut input = EguiInput::new(1.0);
        input.on_touch(7, TouchPhase::Start, (5.0, 5.0));
        input.on_touch(8, TouchPhase::Start, (50.0, 50.0));
        input.on_touch(7, TouchPhase::End, (5.0, 5.0));
        let presses: Vec<bool> = input
            .take([100, 100], 0.0)
            .events
            .iter()
            .filter_map(|e| match e {
                Event::PointerButton { pressed, .. } => Some(*pressed),
                _ => None,
            })
            .collect();
        assert_eq!(presses, vec![true, false]);
    }

    #[test]
    fn bad_scale_factor_falls_back_to_one() {
        let mut input = EguiInput::new(f32::NAN);
        input.on_cursor_moved((10.0, 10.0));
        assert_eq!(
            input.take([1, 1], 0.0).events,
            vec![Event::PointerMoved(Pos2::new(10.0, 10.0))]
        );
    }
}
