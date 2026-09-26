//! Orbit and fly cameras. Right-handed, Y up; yaw 0 / pitch 0 looks down −Z.

use glam::camera::rh::{proj, view};
use glam::{Mat4, Vec2, Vec3};

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

    pub fn projection(aspect: f32) -> Mat4 {
        proj::directx::perspective(FOV_Y, aspect.max(1e-3), 0.1, 500.0)
    }
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
        let view_proj = CameraRig::projection(16.0 / 9.0) * rig.view();
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
    fn pan_moves_target_perpendicular_to_view() {
        let mut o = orbit();
        let (before, fwd) = (o.target, view_forward(o.view()));
        o.pan(0.1, 0.05);
        assert!((o.target - before).dot(fwd).abs() < 1e-4);
        assert!((o.target - before).length() > 0.0);
    }
}
