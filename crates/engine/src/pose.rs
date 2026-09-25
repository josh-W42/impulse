//! Render-side interpolation between the last two physics states.

use glam::{Quat, Vec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub position: Vec3,
    pub orientation: Quat,
}

/// Blends `prev` toward `curr`; `alpha` 0 gives `prev`, 1 gives `curr`.
pub fn interpolate(prev: Pose, curr: Pose, alpha: f32) -> Pose {
    Pose {
        position: prev.position.lerp(curr.position, alpha),
        orientation: prev.orientation.slerp(curr.orientation, alpha),
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_PI_2;

    use super::*;

    fn poses() -> (Pose, Pose) {
        let prev = Pose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        };
        let curr = Pose {
            position: Vec3::new(2.0, 4.0, -6.0),
            orientation: Quat::from_rotation_y(FRAC_PI_2),
        };
        (prev, curr)
    }

    #[test]
    fn endpoints_are_exact() {
        let (prev, curr) = poses();
        assert_eq!(interpolate(prev, curr, 0.0), prev);
        let end = interpolate(prev, curr, 1.0);
        assert!(end.position.abs_diff_eq(curr.position, 1e-6));
        assert!(end.orientation.abs_diff_eq(curr.orientation, 1e-6));
    }

    #[test]
    fn midpoint_is_halfway_in_space_and_angle() {
        let (prev, curr) = poses();
        let mid = interpolate(prev, curr, 0.5);
        assert!(mid.position.abs_diff_eq(Vec3::new(1.0, 2.0, -3.0), 1e-6));
        let expected = Quat::from_rotation_y(FRAC_PI_2 / 2.0);
        assert!(mid.orientation.abs_diff_eq(expected, 1e-6));
    }
}
