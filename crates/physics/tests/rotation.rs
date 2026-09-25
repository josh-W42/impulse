//! Rotation contract tests (no contacts involved).

use glam::{Quat, Vec3};
use impulse_physics::{BodyDesc, Collider, World};

const DT: f32 = 1.0 / 60.0;

#[test]
fn spin_about_a_principal_axis_turns_omega_t_radians() {
    let mut world = World::new(Vec3::ZERO);
    let omega = Vec3::new(0.0, 2.0, 0.0);
    let h = world
        .add_body(
            BodyDesc::dynamic(1.0, Vec3::ZERO)
                .with_collider(Collider::Box {
                    half_extents: Vec3::new(0.5, 1.0, 1.5),
                })
                .with_angular_velocity(omega),
        )
        .unwrap();
    for _ in 0..60 {
        world.step(DT);
    }
    let body = world.body(h).unwrap();
    assert_eq!(body.angular_velocity(), omega, "no torque, no change");
    let expected = Quat::from_rotation_y(2.0);
    // q and -q are the same rotation.
    let error = body.orientation().dot(expected).abs();
    assert!(error > 1.0 - 1e-5, "orientation off by {}", (1.0 - error));
}

#[test]
fn orientation_stays_unit_length() {
    let mut world = World::new(Vec3::ZERO);
    let h = world
        .add_body(
            BodyDesc::dynamic(1.0, Vec3::ZERO)
                .with_collider(Collider::Sphere { radius: 0.5 })
                .with_angular_velocity(Vec3::new(1.0, 2.0, 3.0)),
        )
        .unwrap();
    for _ in 0..10_000 {
        world.step(DT);
    }
    let q = world.body(h).unwrap().orientation();
    assert!((q.length() - 1.0).abs() < 1e-5, "|q| = {}", q.length());
}
