//! Capsule contract tests.

use std::f32::consts::FRAC_PI_2;

use glam::{Quat, Vec3};
use impulse_physics::{BodyDesc, BodyHandle, Collider, Material, World, find_contacts};

const DT: f32 = 1.0 / 60.0;
const RADIUS: f32 = 0.25;
const HALF: f32 = 0.6;

fn world_with_floor() -> World {
    let mut world = World::default();
    world
        .add_body(BodyDesc::fixed(Vec3::ZERO).with_collider(Collider::Plane { normal: Vec3::Y }))
        .unwrap();
    world
}

fn capsule(world: &mut World, position: Vec3, orientation: Quat) -> BodyHandle {
    let desc = BodyDesc {
        orientation,
        ..BodyDesc::dynamic(1.0, position)
    }
    .with_collider(Collider::Capsule {
        half_height: HALF,
        radius: RADIUS,
    })
    .with_material(Material {
        restitution: 0.0,
        friction: 0.6,
    });
    world.add_body(desc).unwrap()
}

fn run(world: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as u32 {
        world.step(DT);
    }
}

/// |axis · up|: 0 when lying flat, 1 when standing.
fn uprightness(world: &World, h: BodyHandle) -> f32 {
    (world.body(h).unwrap().orientation() * Vec3::Y)
        .dot(Vec3::Y)
        .abs()
}

#[test]
fn lying_capsule_rests_on_two_points_without_rolling_away() {
    let mut world = world_with_floor();
    let h = capsule(
        &mut world,
        Vec3::new(0.0, RADIUS + 0.2, 0.0),
        Quat::from_rotation_z(FRAC_PI_2),
    );
    run(&mut world, 3.0);
    let b = world.body(h).unwrap();
    assert!(uprightness(&world, h) < 0.02, "tipped up");
    assert!(
        (b.position().y - RADIUS).abs() < 0.02,
        "rests at {}",
        b.position().y
    );
    assert!(
        Vec3::new(b.position().x, 0.0, b.position().z).length() < 0.05,
        "rolled away"
    );
    let floor_contacts = find_contacts(&world).len();
    assert_eq!(
        floor_contacts, 2,
        "a lying capsule should touch at both ends"
    );
}

#[test]
fn tilted_upright_capsule_topples_and_ends_lying_down() {
    let mut world = world_with_floor();
    let h = capsule(
        &mut world,
        Vec3::new(0.0, HALF + RADIUS + 0.05, 0.0),
        Quat::from_rotation_x(0.2),
    );
    run(&mut world, 5.0);
    assert!(
        uprightness(&world, h) < 0.05,
        "still standing: {}",
        uprightness(&world, h)
    );
    let b = world.body(h).unwrap();
    assert!(
        b.position().y > RADIUS - 0.02 && b.position().y < RADIUS + 0.05,
        "{}",
        b.position().y
    );
}

#[test]
fn capsule_rests_lying_on_top_of_a_box() {
    let mut world = world_with_floor();
    world
        .add_body(
            BodyDesc::fixed(Vec3::new(0.0, 0.5, 0.0)).with_collider(Collider::Box {
                half_extents: Vec3::new(1.0, 0.5, 1.0),
            }),
        )
        .unwrap();
    let h = capsule(
        &mut world,
        Vec3::new(0.0, 1.6, 0.0),
        Quat::from_rotation_z(FRAC_PI_2),
    );
    run(&mut world, 3.0);
    let p = world.body(h).unwrap().position();
    assert!((p.y - (1.0 + RADIUS)).abs() < 0.02, "rests at {p}");
    assert!(uprightness(&world, h) < 0.02);
}

#[test]
fn crossed_capsules_stack() {
    let mut world = world_with_floor();
    let bottom = capsule(
        &mut world,
        Vec3::new(0.0, RADIUS, 0.0),
        Quat::from_rotation_z(FRAC_PI_2),
    );
    let top = capsule(
        &mut world,
        Vec3::new(0.0, 3.0 * RADIUS + 0.3, 0.0),
        Quat::from_rotation_x(FRAC_PI_2),
    );
    run(&mut world, 3.0);
    let (b, t) = (
        world.body(bottom).unwrap().position(),
        world.body(top).unwrap().position(),
    );
    assert!((b.y - RADIUS).abs() < 0.02, "bottom at {b}");
    assert!(
        (t.y - 3.0 * RADIUS).abs() < 0.03,
        "top at {t} (fell through or off?)"
    );
}
