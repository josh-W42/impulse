//! Orbit, fly, strategy and third-person cameras. Right-handed, Y up; yaw 0 /
//! pitch 0 looks down −Z.

use glam::camera::rh::{proj, view};
use glam::{Mat4, Quat, Vec2, Vec3};
use impulse_physics::{Collider, QueryFilter, World};
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

use crate::input::InputState;

const PITCH_LIMIT: f32 = 1.5; // ≈86°, short of straight up/down where look-at degenerates
const MIN_DISTANCE: f32 = 1.0;
const MAX_DISTANCE: f32 = 200.0;
const FOV_Y: f32 = std::f32::consts::FRAC_PI_3;

/// Circles a target point. Positive pitch raises the eye above the target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl OrbitCamera {
    /// Unit vector from the target to the eye.
    fn offset_dir(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cp * sy, sp, cp * cy)
    }

    pub fn eye(&self) -> Vec3 {
        self.target + self.offset_dir() * self.distance
    }

    pub fn rotate(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw += dyaw;
        self.pitch = (self.pitch + dpitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// `factor` < 1 moves closer.
    pub fn zoom(&mut self, factor: f32) {
        if factor.is_finite() && factor > 0.0 {
            self.distance = (self.distance * factor).clamp(MIN_DISTANCE, MAX_DISTANCE);
        }
    }

    /// Slides the target in the view plane; amounts are fractions of the distance.
    pub fn pan(&mut self, right: f32, up: f32) {
        let forward = -self.offset_dir();
        let right_dir = forward.cross(Vec3::Y).normalize();
        let up_dir = right_dir.cross(forward);
        self.target += (right_dir * right + up_dir * up) * self.distance;
    }

    pub fn view(&self) -> Mat4 {
        view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }
}

/// Free camera. Positive pitch looks up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlyCamera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

impl FlyCamera {
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(-sy * cp, sp, -cy * cp)
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    /// Mouse-look: positive `dx` turns right, positive `dy` looks down.
    pub fn look(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx;
        self.pitch = (self.pitch - dy).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    pub fn translate(&mut self, forward: f32, right: f32, up: f32) {
        self.position += self.forward() * forward + self.right() * right + Vec3::Y * up;
    }

    pub fn view(&self) -> Mat4 {
        view::look_to_mat4(self.position, self.forward(), Vec3::Y)
    }
}

/// Limits for [`RtsCamera`]: pitch (radians above horizontal) and distance (m).
const RTS_PITCH: (f32, f32) = (0.35, 1.45);
const RTS_DISTANCE: (f32, f32) = (6.0, 150.0);
/// Keyboard pan speed, in view distances per second.
const RTS_PAN_SPEED: f32 = 0.8;
/// Keyboard rotate and tilt speeds (rad/s).
const RTS_ROTATE_SPEED: f32 = 1.5;
const RTS_TILT_SPEED: f32 = 0.8;
/// Ground moved per pixel of middle-drag or touch drag, in view distances.
const RTS_DRAG_PAN: f32 = 0.0015;

/// Looks down at a point on the ground, like a strategy-game camera. Same
/// spherical convention as [`OrbitCamera`] (yaw 0 looks down −Z).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtsCamera {
    pub focus: Vec3,
    pub yaw: f32,
    /// Radians above horizontal, kept within [0.35, 1.45].
    pub pitch: f32,
    /// Kept within [6, 150].
    pub distance: f32,
    /// Keeps the focus's XZ within (min, max), if set.
    pub bounds: Option<(Vec2, Vec2)>,
}

impl RtsCamera {
    /// Looking at `focus` from 40 m away, about 50° above the horizon.
    pub fn new(focus: Vec3) -> Self {
        Self {
            focus,
            yaw: 0.0,
            pitch: 0.9,
            distance: 40.0,
            bounds: None,
        }
    }

    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.focus + Vec3::new(cp * sy, sp, cp * cy) * self.distance
    }

    pub fn view(&self) -> Mat4 {
        view::look_at_mat4(self.eye(), self.focus, Vec3::Y)
    }

    /// Moves along the ground relative to the view, in metres.
    pub fn pan(&mut self, forward: f32, right: f32) {
        let (sy, cy) = self.yaw.sin_cos();
        let forward_dir = Vec3::new(-sy, 0.0, -cy);
        let right_dir = Vec3::new(cy, 0.0, -sy);
        self.focus += forward_dir * forward + right_dir * right;
        if let Some((min, max)) = self.bounds {
            self.focus.x = self.focus.x.clamp(min.x, max.x);
            self.focus.z = self.focus.z.clamp(min.y, max.y);
        }
    }

    pub fn rotate(&mut self, dyaw: f32) {
        self.yaw += dyaw;
    }

    /// Positive raises the eye toward looking straight down.
    pub fn tilt(&mut self, dpitch: f32) {
        self.pitch = (self.pitch + dpitch).clamp(RTS_PITCH.0, RTS_PITCH.1);
    }

    /// `factor` < 1 moves closer.
    pub fn zoom(&mut self, factor: f32) {
        if factor.is_finite() && factor > 0.0 {
            self.distance = (self.distance * factor).clamp(RTS_DISTANCE.0, RTS_DISTANCE.1);
        }
    }

    /// Default controls: WASD or arrows pan (faster when zoomed out), Q/E
    /// rotate, R/F tilt; while the pointer is free of the UI, the wheel zooms,
    /// middle-drag or a one-finger drag pans, and a pinch zooms.
    pub fn update(&mut self, input: &InputState, dt: f32, pointer_free: bool) {
        let axis = |pos: [KeyCode; 2], neg: [KeyCode; 2]| {
            let held = |keys: [KeyCode; 2]| keys.iter().any(|&k| input.key_held(k));
            held(pos) as i32 as f32 - held(neg) as i32 as f32
        };
        let step = RTS_PAN_SPEED * self.distance * dt;
        self.pan(
            axis(
                [KeyCode::KeyW, KeyCode::ArrowUp],
                [KeyCode::KeyS, KeyCode::ArrowDown],
            ) * step,
            axis(
                [KeyCode::KeyD, KeyCode::ArrowRight],
                [KeyCode::KeyA, KeyCode::ArrowLeft],
            ) * step,
        );
        self.rotate(axis([KeyCode::KeyE; 2], [KeyCode::KeyQ; 2]) * RTS_ROTATE_SPEED * dt);
        self.tilt(axis([KeyCode::KeyR; 2], [KeyCode::KeyF; 2]) * RTS_TILT_SPEED * dt);

        if !pointer_free {
            return;
        }
        if input.wheel() != 0.0 {
            self.zoom(0.9f32.powf(input.wheel()));
        }
        // Dragging moves the ground with the pointer.
        let mut drag = input.touch_rotate();
        if input.button_held(MouseButton::Middle) {
            drag += input.cursor_delta();
        }
        let k = RTS_DRAG_PAN * self.distance;
        self.pan(drag.y * k, -drag.x * k);
        if input.touch_pinch() != 1.0 {
            self.zoom(1.0 / input.touch_pinch());
        }
    }
}

const THIRD_PERSON_PITCH: (f32, f32) = (-0.3, 1.3);
const THIRD_PERSON_DISTANCE: (f32, f32) = (1.5, 25.0);
/// Radians per pixel of right-drag or touch drag.
const THIRD_PERSON_LOOK: f32 = 0.006;
/// Radius of the sphere swept from the target to the eye: keeps the near plane
/// (0.1 m) out of walls.
const THIRD_PERSON_PROBE: f32 = 0.2;
/// Closest the eye gets to the target when something is in the way.
const THIRD_PERSON_MIN_PULL: f32 = 0.3;
/// How fast (1/s) the eye eases back out once the way is clear.
const THIRD_PERSON_EASE_OUT: f32 = 4.0;

/// Follows a character from behind: orbits a target point (usually just above
/// the character's head) and pulls in when a wall or the ground would come
/// between the target and the eye. Same spherical convention as
/// [`OrbitCamera`] (yaw 0 looks down −Z).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThirdPersonCamera {
    pub target: Vec3,
    pub yaw: f32,
    /// Radians above horizontal, kept within [-0.3, 1.3].
    pub pitch: f32,
    /// Wanted distance from the target, kept within [1.5, 25].
    pub distance: f32,
    /// Distance in use after [`ThirdPersonCamera::avoid_obstacles`]: never more
    /// than `distance`.
    current: f32,
}

impl ThirdPersonCamera {
    /// Behind `target` (looking down −Z), 7 m back and 20° above it.
    pub fn new(target: Vec3) -> Self {
        Self {
            target,
            yaw: 0.0,
            pitch: 0.35,
            distance: 7.0,
            current: 7.0,
        }
    }

    /// Unit vector from the target toward the eye.
    fn offset_dir(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cp * sy, sp, cp * cy)
    }

    pub fn eye(&self) -> Vec3 {
        self.target + self.offset_dir() * self.current
    }

    pub fn view(&self) -> Mat4 {
        view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    /// The direction the camera faces, flattened onto the ground: "forward" for
    /// movement keys.
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(-sy, 0.0, -cy)
    }

    /// Screen right, on the ground.
    pub fn right(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(cy, 0.0, -sy)
    }

    /// Positive `dyaw` turns the view left; positive `dpitch` raises the eye.
    pub fn rotate(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw += dyaw;
        self.pitch = (self.pitch + dpitch).clamp(THIRD_PERSON_PITCH.0, THIRD_PERSON_PITCH.1);
    }

    /// `factor` < 1 moves closer.
    pub fn zoom(&mut self, factor: f32) {
        if factor.is_finite() && factor > 0.0 {
            self.distance =
                (self.distance * factor).clamp(THIRD_PERSON_DISTANCE.0, THIRD_PERSON_DISTANCE.1);
            self.current = self.current.min(self.distance);
        }
    }

    /// Default controls while the pointer is free of the UI: right-drag or a
    /// one-finger drag looks around (the view follows the pointer), the wheel
    /// or a pinch zooms.
    pub fn update(&mut self, input: &InputState, pointer_free: bool) {
        if !pointer_free {
            return;
        }
        let mut drag = input.touch_rotate();
        if input.button_held(MouseButton::Right) {
            drag += input.cursor_delta();
        }
        self.rotate(-drag.x * THIRD_PERSON_LOOK, drag.y * THIRD_PERSON_LOOK);
        if input.wheel() != 0.0 {
            self.zoom(0.9f32.powf(input.wheel()));
        }
        if input.touch_pinch() != 1.0 {
            self.zoom(1.0 / input.touch_pinch());
        }
    }

    /// Pulls the eye in front of the first thing (that `filter` accepts)
    /// between the target and the wanted eye position, at once; eases back out
    /// over a fraction of a second when the way clears. Call each frame after
    /// moving the target. Exclude the character's own body in `filter`.
    pub fn avoid_obstacles(&mut self, world: &World, filter: QueryFilter, dt: f32) {
        let probe = Collider::Sphere {
            radius: THIRD_PERSON_PROBE,
        };
        let clear = world
            .shape_cast(
                probe,
                self.target,
                Quat::IDENTITY,
                self.offset_dir(),
                self.distance,
                filter,
            )
            .map_or(self.distance, |hit| hit.distance.max(THIRD_PERSON_MIN_PULL));
        if clear < self.current || !dt.is_finite() {
            self.current = clear;
        } else {
            let ease = 1.0 - (-THIRD_PERSON_EASE_OUT * dt.max(0.0)).exp();
            self.current += (clear - self.current) * ease;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    Orbit,
    Fly,
}

/// Both cameras plus which one is active; switching keeps the view where it was.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraRig {
    pub mode: CameraMode,
    pub orbit: OrbitCamera,
    pub fly: FlyCamera,
}

impl CameraRig {
    pub fn new(orbit: OrbitCamera) -> Self {
        let mut rig = Self {
            mode: CameraMode::Orbit,
            orbit,
            fly: FlyCamera {
                position: Vec3::ZERO,
                yaw: 0.0,
                pitch: 0.0,
            },
        };
        rig.sync_fly_from_orbit();
        rig
    }

    fn sync_fly_from_orbit(&mut self) {
        self.fly = FlyCamera {
            position: self.orbit.eye(),
            yaw: self.orbit.yaw,
            pitch: -self.orbit.pitch,
        };
    }

    pub fn toggle(&mut self) {
        match self.mode {
            CameraMode::Orbit => {
                self.sync_fly_from_orbit();
                self.mode = CameraMode::Fly;
            }
            CameraMode::Fly => {
                self.orbit.yaw = self.fly.yaw;
                self.orbit.pitch = -self.fly.pitch;
                self.orbit.target = self.fly.position + self.fly.forward() * self.orbit.distance;
                self.mode = CameraMode::Orbit;
            }
        }
    }

    pub fn eye(&self) -> Vec3 {
        match self.mode {
            CameraMode::Orbit => self.orbit.eye(),
            CameraMode::Fly => self.fly.position,
        }
    }

    pub fn view(&self) -> Mat4 {
        match self.mode {
            CameraMode::Orbit => self.orbit.view(),
            CameraMode::Fly => self.fly.view(),
        }
    }
}

/// The engine's perspective projection (60° vertical field of view) for a
/// render target of the given width over height.
pub fn perspective(aspect: f32) -> Mat4 {
    proj::directx::perspective(FOV_Y, aspect.max(1e-3), 0.1, 500.0)
}

/// World-space ray `(origin on the near plane, unit direction)` through a point in
/// normalized device coordinates (x right, y up, both in [-1, 1]).
pub fn cursor_ray(view_proj: Mat4, ndc: Vec2) -> (Vec3, Vec3) {
    let inverse = view_proj.inverse();
    let near = inverse.project_point3(Vec3::new(ndc.x, ndc.y, 0.0));
    let far = inverse.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
    (near, (far - near).normalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orbit() -> OrbitCamera {
        OrbitCamera {
            target: Vec3::new(1.0, 2.0, 3.0),
            yaw: 0.7,
            pitch: 0.4,
            distance: 20.0,
        }
    }

    /// Direction the view matrix looks along (world space).
    fn view_forward(view: Mat4) -> Vec3 {
        view.inverse().transform_vector3(Vec3::NEG_Z).normalize()
    }

    #[test]
    fn screen_center_ray_leaves_the_eye_along_the_view() {
        let rig = CameraRig::new(orbit());
        let view_proj = perspective(16.0 / 9.0) * rig.view();
        let (origin, dir) = cursor_ray(view_proj, Vec2::ZERO);
        assert!(dir.abs_diff_eq(view_forward(rig.view()), 1e-4), "{dir}");
        assert!(origin.distance(rig.eye()) < 0.2, "starts at the near plane");
        // A point up-right on screen sends the ray up and right of center.
        let (_, corner) = cursor_ray(view_proj, Vec2::new(0.5, 0.5));
        let right = view_forward(rig.view()).cross(Vec3::Y);
        assert!(corner.dot(right) > 0.0 && corner.y > dir.y);
    }

    #[test]
    fn pitch_stays_clamped() {
        let mut o = orbit();
        o.rotate(0.0, 100.0);
        assert!(o.pitch <= PITCH_LIMIT);
        o.rotate(0.0, -100.0);
        assert!(o.pitch >= -PITCH_LIMIT);
        let mut f = FlyCamera {
            position: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
        };
        f.look(0.0, 100.0);
        assert!(f.pitch >= -PITCH_LIMIT);
    }

    #[test]
    fn zoom_stays_clamped_and_ignores_bad_factors() {
        let mut o = orbit();
        o.zoom(0.0001);
        assert_eq!(o.distance, MIN_DISTANCE);
        o.zoom(1e6);
        assert_eq!(o.distance, MAX_DISTANCE);
        o.zoom(f32::NAN);
        o.zoom(-2.0);
        assert_eq!(o.distance, MAX_DISTANCE);
    }

    #[test]
    fn orbit_looks_at_its_target() {
        let o = orbit();
        let expected = (o.target - o.eye()).normalize();
        assert!(view_forward(o.view()).abs_diff_eq(expected, 1e-4));
    }

    #[test]
    fn switching_to_fly_keeps_the_view() {
        let mut rig = CameraRig::new(orbit());
        let (eye, fwd) = (rig.eye(), view_forward(rig.view()));
        rig.toggle();
        assert_eq!(rig.mode, CameraMode::Fly);
        assert!(rig.eye().abs_diff_eq(eye, 1e-4));
        assert!(view_forward(rig.view()).abs_diff_eq(fwd, 1e-4));
    }

    #[test]
    fn fly_round_trip_keeps_position() {
        let mut rig = CameraRig::new(orbit());
        rig.toggle();
        rig.fly.translate(3.0, -2.0, 1.0);
        rig.fly.look(0.3, -0.2);
        let (eye, fwd) = (rig.eye(), view_forward(rig.view()));
        rig.toggle();
        rig.toggle();
        assert!(rig.eye().abs_diff_eq(eye, 1e-3));
        assert!(view_forward(rig.view()).abs_diff_eq(fwd, 1e-4));
    }

    #[test]
    fn fly_moves_along_its_view_direction() {
        let mut f = FlyCamera {
            position: Vec3::ZERO,
            yaw: 0.4,
            pitch: 0.3,
        };
        let fwd = view_forward(f.view());
        f.translate(1.0, 0.0, 0.0);
        assert!(f.position.abs_diff_eq(fwd, 1e-4));
    }

    #[test]
    fn third_person_starts_behind_the_target() {
        let cam = ThirdPersonCamera::new(Vec3::new(0.0, 1.6, 0.0));
        let eye = cam.eye();
        assert!(eye.z > 6.0 && eye.y > 1.6, "{eye}");
        assert!(view_forward(cam.view()).dot(Vec3::NEG_Z) > 0.9);
        assert!(cam.forward().abs_diff_eq(Vec3::NEG_Z, 1e-6));
        assert!(cam.right().abs_diff_eq(Vec3::X, 1e-6));
    }

    #[test]
    fn third_person_forward_matches_the_view() {
        let mut cam = ThirdPersonCamera::new(Vec3::ZERO);
        cam.rotate(1.1, 0.4);
        let seen = view_forward(cam.view());
        let flat = Vec3::new(seen.x, 0.0, seen.z).normalize();
        assert!(
            cam.forward().abs_diff_eq(flat, 1e-4),
            "{} vs {flat}",
            cam.forward()
        );
        assert!(cam.right().abs_diff_eq(cam.forward().cross(Vec3::Y), 1e-6));
    }

    #[test]
    fn third_person_pitch_and_zoom_are_clamped() {
        let mut cam = ThirdPersonCamera::new(Vec3::ZERO);
        cam.rotate(0.0, 10.0);
        assert_eq!(cam.pitch, THIRD_PERSON_PITCH.1);
        cam.rotate(0.0, -20.0);
        assert_eq!(cam.pitch, THIRD_PERSON_PITCH.0);
        cam.zoom(0.0001);
        assert_eq!(cam.distance, THIRD_PERSON_DISTANCE.0);
        cam.zoom(1e6);
        assert_eq!(cam.distance, THIRD_PERSON_DISTANCE.1);
        assert!(cam.current <= cam.distance);
    }

    #[test]
    fn third_person_pulls_in_at_walls_and_eases_back_out() {
        use impulse_physics::BodyDesc;

        let mut world = World::default();
        // A wall 3 m behind the target (the eye wants to be 7 m back).
        let wall = world
            .add_body(
                BodyDesc::fixed(Vec3::new(0.0, 2.0, 3.5)).with_collider(Collider::Box {
                    half_extents: Vec3::new(5.0, 3.0, 0.5),
                }),
            )
            .unwrap();
        let mut cam = ThirdPersonCamera {
            pitch: 0.0,
            ..ThirdPersonCamera::new(Vec3::new(0.0, 1.6, 0.0))
        };
        cam.avoid_obstacles(&world, QueryFilter::ALL, 1.0 / 60.0);
        // The probe sphere stops at the wall face (z = 3): eye 0.2 m in front.
        assert!((cam.eye().z - 2.8).abs() < 0.01, "{}", cam.eye());
        world.remove_body(wall);
        cam.avoid_obstacles(&world, QueryFilter::ALL, 1.0 / 60.0);
        let after_one = cam.eye().z;
        assert!(
            after_one > 2.8 && after_one < 4.0,
            "eases out, not jumps: {after_one}"
        );
        for _ in 0..120 {
            cam.avoid_obstacles(&world, QueryFilter::ALL, 1.0 / 60.0);
        }
        assert!((cam.eye().z - 7.0).abs() < 0.01, "{}", cam.eye());
    }

    fn rts() -> RtsCamera {
        RtsCamera {
            yaw: 0.7,
            ..RtsCamera::new(Vec3::new(64.0, 0.0, 64.0))
        }
    }

    fn horizontal(v: Vec3) -> Vec2 {
        Vec2::new(v.x, v.z)
    }

    #[test]
    fn rts_pan_forward_moves_focus_away_from_the_eye() {
        let mut cam = rts();
        let (eye, before) = (cam.eye(), cam.focus);
        cam.pan(5.0, 0.0);
        let gap = |focus: Vec3| horizontal(focus).distance(horizontal(eye));
        assert!((gap(cam.focus) - gap(before) - 5.0).abs() < 1e-3);
    }

    #[test]
    fn rts_pan_right_moves_focus_to_screen_right() {
        let mut cam = rts();
        let view_proj = perspective(1.5) * cam.view();
        cam.pan(0.0, 5.0);
        let ndc = view_proj.project_point3(cam.focus);
        assert!(ndc.x > 0.1 && ndc.y.abs() < 1e-3, "{ndc}");
    }

    #[test]
    fn rts_focus_stays_on_the_ground_and_in_bounds() {
        let mut cam = RtsCamera {
            bounds: Some((Vec2::ZERO, Vec2::splat(128.0))),
            ..rts()
        };
        cam.pan(1000.0, -1000.0);
        assert_eq!(cam.focus.y, 0.0);
        for v in [cam.focus.x, cam.focus.z] {
            assert!((0.0..=128.0).contains(&v), "{}", cam.focus);
        }
    }

    #[test]
    fn rts_zoom_and_tilt_are_clamped() {
        let mut cam = rts();
        cam.zoom(1e6);
        assert_eq!(cam.distance, RTS_DISTANCE.1);
        cam.zoom(1e-6);
        assert_eq!(cam.distance, RTS_DISTANCE.0);
        cam.zoom(f32::NAN);
        cam.zoom(-1.0);
        assert_eq!(cam.distance, RTS_DISTANCE.0);
        cam.tilt(100.0);
        assert_eq!(cam.pitch, RTS_PITCH.1);
        cam.tilt(-100.0);
        assert_eq!(cam.pitch, RTS_PITCH.0);
    }

    #[test]
    fn rts_center_ray_hits_the_focus() {
        let cam = rts();
        let (origin, dir) = cursor_ray(perspective(1.5) * cam.view(), Vec2::ZERO);
        let to_focus = cam.focus - origin;
        let miss = to_focus - dir * to_focus.dot(dir);
        // f32 rounding through the inverse projection (near 0.1, far 500) costs
        // a few millimetres at 40 m; a convention error would miss by metres.
        assert!(miss.length() < 0.01, "{miss}");
    }

    #[test]
    fn rts_keys_pan_relative_to_the_view() {
        let mut cam = rts();
        let view_proj = perspective(1.5) * cam.view();
        let mut input = InputState::new();
        input.on_key(KeyCode::KeyD, true);
        cam.update(&input, 0.5, true);
        assert!(view_proj.project_point3(cam.focus).x > 0.1);
    }

    #[test]
    fn rts_wheel_zooms_only_while_the_pointer_is_free() {
        let mut cam = rts();
        let mut input = InputState::new();
        input.on_wheel(2.0);
        let before = cam.distance;
        cam.update(&input, 0.016, false);
        assert_eq!(cam.distance, before, "egui has the pointer");
        cam.update(&input, 0.016, true);
        assert!(cam.distance < before, "wheel up zooms in");
    }

    #[test]
    fn pan_moves_target_perpendicular_to_view() {
        let mut o = orbit();
        let (before, fwd) = (o.target, view_forward(o.view()));
        o.pan(0.1, 0.05);
        assert!((o.target - before).dot(fwd).abs() < 1e-4);
        assert!((o.target - before).length() > 0.0);
    }
}
