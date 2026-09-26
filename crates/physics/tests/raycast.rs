//! Raycast contract tests against analytic answers.

use std::f32::consts::FRAC_PI_4;

use glam::{Quat, Vec3};
use impulse_physics::{BodyDesc, Collider, World};

fn add(
    world: &mut World,
    collider: Collider,
    position: Vec3,
    orientation: Quat,
) -> impulse_physics::BodyHandle {
    world
        .add_body(
            BodyDesc {
                orientation,
                ..BodyDesc::fixed(position)
            }
            .with_collider(collider),
        )
        .unwrap()
}

#[test]
fn hits_a_sphere_at_the_analytic_distance() {
    let mut world = World::new(Vec3::ZERO);
    let s = add(
        &mut world,
        Collider::Sphere { radius: 1.0 },
        Vec3::new(5.0, 0.0, 0.0),
        Quat::IDENTITY,
    );
    let hit = world.raycast(Vec3::ZERO, Vec3::X * 3.0, 100.0).unwrap();
    assert_eq!(hit.body, s);
    assert!((hit.t - 4.0).abs() < 1e-5 && hit.normal.abs_diff_eq(Vec3::NEG_X, 1e-5));
    assert!(world.raycast(Vec3::ZERO, Vec3::Y, 100.0).is_none(), "miss");
    assert!(
        world.raycast(Vec3::ZERO, Vec3::X, 3.9).is_none(),
        "beyond max distance"
    );
}

#[test]
fn hits_a_rotated_box_on_its_turned_face() {
    let mut world = World::new(Vec3::ZERO);
    let turned = Quat::from_rotation_y(FRAC_PI_4);
    add(
        &mut world,
        Collider::Box {
            half_extents: Vec3::splat(0.5),
        },
        Vec3::new(0.0, 0.0, -5.0),
        turned,
    );
    let hit = world.raycast(Vec3::ZERO, Vec3::NEG_Z, 100.0).unwrap();
    // Turned 45°, the near corner points at the ray: 5 − √2·0.5.
    assert!(
        (hit.t - (5.0 - 0.5 * 2f32.sqrt())).abs() < 1e-4,
        "{}",
        hit.t
    );
    assert!(hit.normal.dot(Vec3::Z) > 0.5, "{}", hit.normal);
}

#[test]
fn hits_planes_only_from_the_front() {
    let mut world = World::new(Vec3::ZERO);
    add(
        &mut world,
        Collider::Plane { normal: Vec3::Y },
        Vec3::ZERO,
        Quat::IDENTITY,
    );
    let hit = world
        .raycast(Vec3::new(0.0, 2.0, 0.0), Vec3::new(1.0, -1.0, 0.0), 100.0)
        .unwrap();
    assert!((hit.t - 2.0 * 2f32.sqrt()).abs() < 1e-4);
    assert!(
        world
            .raycast(Vec3::new(0.0, -2.0, 0.0), Vec3::Y, 100.0)
            .is_none(),
        "from behind"
    );
    assert!(
        world
            .raycast(Vec3::new(0.0, 2.0, 0.0), Vec3::X, 100.0)
            .is_none(),
        "parallel"
    );
}

#[test]
fn hits_a_capsule_on_its_side_and_its_cap() {
    let mut world = World::new(Vec3::ZERO);
    add(
        &mut world,
        Collider::Capsule {
            half_height: 1.0,
            radius: 0.5,
        },
        Vec3::ZERO,
        Quat::IDENTITY,
    );
    let side = world
        .raycast(Vec3::new(-5.0, 0.5, 0.0), Vec3::X, 100.0)
        .unwrap();
    assert!((side.t - 4.5).abs() < 1e-4 && side.normal.abs_diff_eq(Vec3::NEG_X, 1e-4));
    let cap = world
        .raycast(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y, 100.0)
        .unwrap();
    assert!((cap.t - 3.5).abs() < 1e-4 && cap.normal.abs_diff_eq(Vec3::Y, 1e-4));
}

#[test]
fn returns_the_nearest_of_several_hits_including_dynamic_bodies() {
    let mut world = World::default();
    let far = add(
        &mut world,
        Collider::Sphere { radius: 0.5 },
        Vec3::new(0.0, 0.0, -9.0),
        Quat::IDENTITY,
    );
    let near = world
        .add_body(
            BodyDesc::dynamic(1.0, Vec3::new(0.0, 0.0, -4.0))
                .with_collider(Collider::Sphere { radius: 0.5 }),
        )
        .unwrap();
    assert_eq!(
        world.raycast(Vec3::ZERO, Vec3::NEG_Z, 100.0).unwrap().body,
        near
    );
    world.remove_body(near);
    assert_eq!(
        world.raycast(Vec3::ZERO, Vec3::NEG_Z, 100.0).unwrap().body,
        far
    );
}

#[test]
fn ray_starting_inside_a_shape_ignores_it() {
    let mut world = World::new(Vec3::ZERO);
    add(
        &mut world,
        Collider::Sphere { radius: 2.0 },
        Vec3::ZERO,
        Quat::IDENTITY,
    );
    assert!(world.raycast(Vec3::ZERO, Vec3::X, 100.0).is_none());
}
