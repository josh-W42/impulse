//! Kinematic bodies: moved only by the velocity they're given, they push
//! dynamic bodies and are never pushed back.

use glam::{Quat, Vec3};
use impulse_physics::{BodyDesc, BodyError, BodyHandle, Collider, Material, World};

const DT: f32 = 1.0 / 60.0;

fn world_with_floor() -> World {
    let mut world = World::default();
    world
        .add_body(BodyDesc::fixed(Vec3::ZERO).with_collider(Collider::Plane { normal: Vec3::Y }))
        .unwrap();
    world
}

fn crate_at(world: &mut World, position: Vec3) -> BodyHandle {
    world
        .add_body(
            BodyDesc::dynamic(1.0, position)
                .with_collider(Collider::Box {
                    half_extents: Vec3::splat(0.5),
                })
                .with_material(Material {
                    restitution: 0.0,
                    friction: 0.8,
                }),
        )
        .unwrap()
}

fn run(world: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as u32 {
        world.step(DT);
    }
}

fn position(world: &World, h: BodyHandle) -> Vec3 {
    world.body(h).unwrap().position()
}

#[test]
fn moves_with_its_velocity_and_ignores_gravity() {
    let mut world = World::default();
    let k = world
        .add_body(
            BodyDesc::kinematic(Vec3::new(0.0, 5.0, 0.0)).with_velocity(Vec3::new(2.0, 0.0, 0.0)),
        )
        .unwrap();
    run(&mut world, 1.0);
    assert!(position(&world, k).abs_diff_eq(Vec3::new(2.0, 5.0, 0.0), 1e-4));
    assert_eq!(world.body(k).unwrap().inverse_mass(), 0.0);
}

#[test]
fn is_not_pushed_by_what_lands_on_it() {
    let mut world = World::default();
    let platform = world
        .add_body(
            BodyDesc::kinematic(Vec3::ZERO).with_collider(Collider::Box {
                half_extents: Vec3::new(3.0, 0.25, 3.0),
            }),
        )
        .unwrap();
    let heavy = world
        .add_body(
            BodyDesc::dynamic(1000.0, Vec3::new(0.0, 3.0, 0.0))
                .with_collider(Collider::Sphere { radius: 0.5 }),
        )
        .unwrap();
    run(&mut world, 2.0);
    assert_eq!(position(&world, platform), Vec3::ZERO);
    assert!(
        (position(&world, heavy).y - 0.75).abs() < 0.05,
        "{}",
        position(&world, heavy)
    );
}

#[test]
fn kinematic_target_is_reached_in_one_step() {
    let mut world = World::default();
    let k = world.add_body(BodyDesc::kinematic(Vec3::ZERO)).unwrap();
    let target = Vec3::new(1.0, -2.0, 0.5);
    let turn = Quat::from_rotation_y(0.4) * Quat::from_rotation_x(-0.2);
    assert!(world.set_kinematic_target(k, target, turn, DT));
    world.step(DT);
    let body = world.body(k).unwrap();
    assert!(
        body.position().abs_diff_eq(target, 1e-4),
        "{}",
        body.position()
    );
    assert!(body.orientation().angle_between(turn) < 1e-3);
}

#[test]
fn kinematic_target_turns_the_short_way() {
    let mut world = World::default();
    let k = world
        .add_body(BodyDesc {
            orientation: Quat::from_rotation_y(3.0),
            ..BodyDesc::kinematic(Vec3::ZERO)
        })
        .unwrap();
    // 3.0 → -3.0 rad is 0.28 rad through ±π, not 6 rad back the other way.
    world.set_kinematic_target(k, Vec3::ZERO, Quat::from_rotation_y(-3.0), DT);
    let spin = world.body(k).unwrap().angular_velocity().length() * DT;
    assert!(
        (spin - (2.0 * std::f32::consts::PI - 6.0)).abs() < 1e-3,
        "turned {spin} rad"
    );
}

#[test]
fn kinematic_target_only_moves_kinematic_bodies() {
    let mut world = World::default();
    let d = world.add_body(BodyDesc::dynamic(1.0, Vec3::ZERO)).unwrap();
    let s = world.add_body(BodyDesc::fixed(Vec3::ZERO)).unwrap();
    let k = world.add_body(BodyDesc::kinematic(Vec3::ZERO)).unwrap();
    assert!(!world.set_kinematic_target(d, Vec3::X, Quat::IDENTITY, DT));
    assert!(!world.set_kinematic_target(s, Vec3::X, Quat::IDENTITY, DT));
    assert!(!world.set_kinematic_target(k, Vec3::NAN, Quat::IDENTITY, DT));
    assert!(!world.set_kinematic_target(k, Vec3::X, Quat::IDENTITY, 0.0));
}

#[test]
fn pushes_dynamic_bodies_out_of_its_way() {
    let mut world = world_with_floor();
    let block = crate_at(&mut world, Vec3::new(0.0, 0.5, 0.0));
    let wall = world
        .add_body(
            BodyDesc::kinematic(Vec3::new(-2.0, 0.5, 0.0))
                .with_collider(Collider::Box {
                    half_extents: Vec3::new(0.25, 0.5, 1.0),
                })
                .with_velocity(Vec3::new(1.5, 0.0, 0.0)),
        )
        .unwrap();
    run(&mut world, 3.0);
    let (wall_x, block_x) = (position(&world, wall).x, position(&world, block).x);
    assert!(
        (wall_x - 2.5).abs() < 1e-3,
        "the wall keeps its speed: {wall_x}"
    );
    // Pushed ahead of the wall, in contact with its face (0.25 + 0.5 apart).
    assert!(
        block_x > wall_x + 0.7,
        "block {block_x} behind wall {wall_x}"
    );
    assert!(
        block_x < wall_x + 0.85,
        "block {block_x} not pushed by wall {wall_x}"
    );
}

#[test]
fn carries_what_rests_on_it() {
    let mut world = World::default();
    let platform = world
        .add_body(
            BodyDesc::kinematic(Vec3::ZERO).with_collider(Collider::Box {
                half_extents: Vec3::new(2.0, 0.25, 2.0),
            }),
        )
        .unwrap();
    let rider = crate_at(&mut world, Vec3::new(0.0, 0.75, 0.0));
    run(&mut world, 1.0); // settle
    let start = position(&world, rider);
    world.set_velocity(platform, Vec3::new(1.0, 0.0, 0.0));
    run(&mut world, 1.0);
    let moved = position(&world, rider) - start;
    assert!((moved.x - 1.0).abs() < 0.1, "rider moved {moved}");
    assert!(moved.y.abs() < 0.02, "rider moved {moved}");
}

#[test]
fn wakes_sleeping_bodies_it_touches() {
    let mut world = world_with_floor();
    let block = crate_at(&mut world, Vec3::new(0.0, 0.5, 0.0));
    let pusher = world
        .add_body(
            BodyDesc::kinematic(Vec3::new(-1.5, 0.5, 0.0))
                .with_collider(Collider::Sphere { radius: 0.5 }),
        )
        .unwrap();
    run(&mut world, 2.0);
    assert!(world.body(block).unwrap().is_asleep());
    world.set_velocity(pusher, Vec3::new(1.0, 0.0, 0.0));
    run(&mut world, 1.0);
    assert!(
        position(&world, block).x > 0.1,
        "sleeping block wasn't pushed"
    );
}

#[test]
fn passes_through_static_and_kinematic_bodies() {
    let mut world = world_with_floor();
    world
        .add_body(
            BodyDesc::fixed(Vec3::new(2.0, 0.5, 0.0)).with_collider(Collider::Box {
                half_extents: Vec3::splat(0.5),
            }),
        )
        .unwrap();
    world
        .add_body(
            BodyDesc::kinematic(Vec3::new(4.0, 0.5, 0.0)).with_collider(Collider::Box {
                half_extents: Vec3::splat(0.5),
            }),
        )
        .unwrap();
    let k = world
        .add_body(
            BodyDesc::kinematic(Vec3::new(0.0, 0.25, 0.0))
                .with_collider(Collider::Sphere { radius: 0.5 })
                .with_velocity(Vec3::new(3.0, 0.0, 0.0)),
        )
        .unwrap();
    for _ in 0..120 {
        let stats = world.step_profiled(DT, &mut || 0.0);
        assert_eq!(
            stats.contacts, 0,
            "kinematic bodies don't collide with these"
        );
    }
    assert!(position(&world, k).abs_diff_eq(Vec3::new(6.0, 0.25, 0.0), 1e-3));
}

#[test]
fn rejects_kinematic_planes() {
    let mut world = World::default();
    let plane = BodyDesc::kinematic(Vec3::ZERO).with_collider(Collider::Plane { normal: Vec3::Y });
    assert_eq!(world.add_body(plane), Err(BodyError::InvalidCollider));
}

#[test]
fn set_pose_teleports_and_wakes() {
    let mut world = world_with_floor();
    let block = crate_at(&mut world, Vec3::new(0.0, 0.5, 0.0));
    run(&mut world, 2.0);
    assert!(world.body(block).unwrap().is_asleep());
    assert!(world.set_pose(block, Vec3::new(10.0, 3.0, 0.0), Quat::IDENTITY));
    assert!(!world.body(block).unwrap().is_asleep());
    // The broadphase knows about the move straight away.
    let hit = world
        .raycast(Vec3::new(10.0, 10.0, 0.0), Vec3::NEG_Y, 20.0)
        .unwrap();
    assert_eq!(hit.body, block);
    run(&mut world, 2.0);
    assert!(
        (position(&world, block).y - 0.5).abs() < 0.02,
        "fell back to the floor"
    );
    assert!(!world.set_pose(block, Vec3::INFINITY, Quat::IDENTITY));
}

#[test]
fn set_pose_wakes_what_rested_on_the_moved_body() {
    let mut world = World::default();
    let shelf = world
        .add_body(BodyDesc::fixed(Vec3::ZERO).with_collider(Collider::Box {
            half_extents: Vec3::new(2.0, 0.25, 2.0),
        }))
        .unwrap();
    let block = crate_at(&mut world, Vec3::new(0.0, 0.75, 0.0));
    run(&mut world, 2.0);
    assert!(world.body(block).unwrap().is_asleep());
    world.set_pose(shelf, Vec3::new(0.0, -50.0, 0.0), Quat::IDENTITY);
    run(&mut world, 0.5);
    assert!(
        position(&world, block).y < 0.0,
        "the block should fall once the shelf is gone"
    );
}

#[test]
fn set_velocity_rules() {
    let mut world = World::default();
    let s = world.add_body(BodyDesc::fixed(Vec3::ZERO)).unwrap();
    let d = world.add_body(BodyDesc::dynamic(1.0, Vec3::ZERO)).unwrap();
    assert!(!world.set_velocity(s, Vec3::X), "static bodies don't move");
    assert!(world.set_velocity(d, Vec3::X));
    assert!(!world.set_velocity(d, Vec3::NAN));
    assert_eq!(world.body(d).unwrap().velocity(), Vec3::X);
    // No collider, so no inertia: a dynamic body that can't rotate ignores spin.
    assert!(world.set_angular_velocity(d, Vec3::Y));
    assert_eq!(world.body(d).unwrap().angular_velocity(), Vec3::ZERO);
}
