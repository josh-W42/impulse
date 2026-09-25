//! Box contract tests against a floor.

use glam::{Quat, Vec3};
use impulse_physics::{BodyDesc, BodyHandle, Collider, Material, World};

const DT: f32 = 1.0 / 60.0;

fn world_with_floor() -> World {
    let mut world = World::default();
    world
        .add_body(BodyDesc::fixed(Vec3::ZERO).with_collider(Collider::Plane { normal: Vec3::Y }))
        .unwrap();
    world
}

fn drop_box(
    world: &mut World,
    half_extents: Vec3,
    position: Vec3,
    orientation: Quat,
) -> BodyHandle {
    let desc = BodyDesc {
        orientation,
        ..BodyDesc::dynamic(1.0, position)
            .with_collider(Collider::Box { half_extents })
            .with_material(Material {
                restitution: 0.0,
                friction: 0.6,
            })
    };
    world.add_body(desc).unwrap()
}

fn run(world: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as u32 {
        world.step(DT);
    }
}

/// Smallest angle (degrees) between world up and any of the body's ± axes.
fn tilt_degrees(orientation: Quat) -> f32 {
    [Vec3::X, Vec3::Y, Vec3::Z]
        .map(|axis| (orientation * axis).dot(Vec3::Y).abs())
        .into_iter()
        .fold(0.0, f32::max)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

#[test]
fn box_dropped_flat_comes_to_rest_flat() {
    let mut world = world_with_floor();
    let h = drop_box(
        &mut world,
        Vec3::new(0.5, 0.3, 0.4),
        Vec3::new(0.0, 2.0, 0.0),
        Quat::IDENTITY,
    );
    run(&mut world, 4.0);
    let b = world.body(h).unwrap();
    assert!(
        tilt_degrees(b.orientation()) < 1.0,
        "tilted {}°",
        tilt_degrees(b.orientation())
    );
    assert!(
        b.velocity().length() < 0.05 && b.angular_velocity().length() < 0.05,
        "still moving"
    );
    assert!(
        (b.position().y - 0.3).abs() < 0.02,
        "rests at {}",
        b.position().y
    );
}

#[test]
fn tilted_box_topples_and_settles_on_a_face() {
    let mut world = world_with_floor();
    let tilt = Quat::from_rotation_z(0.6) * Quat::from_rotation_x(0.3);
    let h = drop_box(
        &mut world,
        Vec3::new(0.25, 0.8, 0.4),
        Vec3::new(0.0, 1.5, 0.0),
        tilt,
    );
    run(&mut world, 6.0);
    let b = world.body(h).unwrap();
    assert!(
        tilt_degrees(b.orientation()) < 5.0,
        "not on a face: {}°",
        tilt_degrees(b.orientation())
    );
    assert!(
        b.velocity().length() < 0.1 && b.angular_velocity().length() < 0.1,
        "still moving"
    );
    assert!(b.position().y > 0.2, "sank: {}", b.position().y);
}

fn world_with_block() -> World {
    let mut world = world_with_floor();
    world
        .add_body(
            BodyDesc::fixed(Vec3::new(0.0, 0.5, 0.0)).with_collider(Collider::Box {
                half_extents: Vec3::splat(0.5),
            }),
        )
        .unwrap();
    world
}

fn add_ball(world: &mut World, position: Vec3, velocity: Vec3) -> BodyHandle {
    world
        .add_body(
            BodyDesc::dynamic(1.0, position)
                .with_collider(Collider::Sphere { radius: 0.3 })
                .with_velocity(velocity),
        )
        .unwrap()
}

#[test]
fn ball_dropped_on_a_box_rests_on_its_top_face() {
    let mut world = world_with_block();
    let ball = add_ball(&mut world, Vec3::new(0.2, 2.0, 0.0), Vec3::ZERO);
    run(&mut world, 3.0);
    let p = world.body(ball).unwrap().position();
    assert!((p.y - 1.3).abs() < 0.02 && (p.x - 0.2).abs() < 0.01, "{p}");
}

#[test]
fn ball_rolls_off_the_edge_of_a_box() {
    let mut world = world_with_block();
    let ball = add_ball(
        &mut world,
        Vec3::new(0.0, 1.3, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
    );
    run(&mut world, 3.0);
    let p = world.body(ball).unwrap().position();
    assert!(
        p.is_finite() && p.x > 0.8 && (p.y - 0.3).abs() < 0.05,
        "should end on the floor: {p}"
    );
}
