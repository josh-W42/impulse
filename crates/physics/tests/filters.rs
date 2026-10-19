//! Collision filters: groups and masks decide which bodies collide, and query
//! filters decide which bodies raycasts see.

use glam::Vec3;
use impulse_physics::{BodyDesc, BodyHandle, Collider, CollisionFilter, QueryFilter, World};

const DT: f32 = 1.0 / 60.0;

const WORLD: u32 = 1 << 0;
const PLAYER: u32 = 1 << 1;
const GHOST: u32 = 1 << 2;

fn run(world: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as u32 {
        world.step(DT);
    }
}

fn floor(world: &mut World) -> BodyHandle {
    world
        .add_body(
            BodyDesc::fixed(Vec3::ZERO)
                .with_collider(Collider::Plane { normal: Vec3::Y })
                .with_filter(CollisionFilter::new(WORLD, u32::MAX)),
        )
        .unwrap()
}

fn ball(world: &mut World, position: Vec3, filter: CollisionFilter) -> BodyHandle {
    world
        .add_body(
            BodyDesc::dynamic(1.0, position)
                .with_collider(Collider::Sphere { radius: 0.5 })
                .with_filter(filter),
        )
        .unwrap()
}

fn height(world: &World, h: BodyHandle) -> f32 {
    world.body(h).unwrap().position().y
}

#[test]
fn bodies_whose_masks_exclude_each_other_pass_through() {
    let mut world = World::default();
    floor(&mut world);
    // Players collide with the world and with each other, not with ghosts.
    let player = CollisionFilter::new(PLAYER, WORLD | PLAYER);
    let ghost = CollisionFilter::new(GHOST, WORLD);
    let lower = ball(&mut world, Vec3::new(0.0, 0.5, 0.0), player);
    let falling_ghost = ball(&mut world, Vec3::new(0.0, 3.0, 0.0), ghost);
    let falling_player = ball(&mut world, Vec3::new(5.0, 3.0, 0.0), player);
    ball(&mut world, Vec3::new(5.0, 0.5, 0.0), player);
    run(&mut world, 2.0);
    assert!(
        (height(&world, falling_ghost) - 0.5).abs() < 0.02,
        "the ghost falls through the player onto the floor"
    );
    assert!(
        (height(&world, falling_player) - 1.5).abs() < 0.05,
        "the player lands on the other player"
    );
    assert!((height(&world, lower) - 0.5).abs() < 0.02);
}

#[test]
fn one_sided_masks_do_not_collide() {
    let mut world = World::default();
    floor(&mut world);
    // The upper ball wants to hit players, but players don't accept it.
    let lower = ball(
        &mut world,
        Vec3::new(0.0, 0.5, 0.0),
        CollisionFilter::new(PLAYER, WORLD),
    );
    let upper = ball(
        &mut world,
        Vec3::new(0.0, 3.0, 0.0),
        CollisionFilter::new(GHOST, WORLD | PLAYER),
    );
    run(&mut world, 2.0);
    assert!((height(&world, upper) - 0.5).abs() < 0.02);
    assert!((height(&world, lower) - 0.5).abs() < 0.02);
}

#[test]
fn planes_respect_filters() {
    let mut world = World::default();
    floor(&mut world);
    let not_world = ball(
        &mut world,
        Vec3::new(0.0, 1.0, 0.0),
        CollisionFilter::new(GHOST, GHOST),
    );
    run(&mut world, 1.0);
    assert!(height(&world, not_world) < -1.0, "fell through the floor");
}

#[test]
fn changing_a_filter_takes_effect_and_wakes() {
    let mut world = World::default();
    floor(&mut world);
    let b = ball(
        &mut world,
        Vec3::new(0.0, 0.5, 0.0),
        CollisionFilter::DEFAULT,
    );
    run(&mut world, 2.0);
    assert!(world.body(b).unwrap().is_asleep());
    assert!(world.set_filter(b, CollisionFilter::new(GHOST, GHOST)));
    assert!(!world.body(b).unwrap().is_asleep());
    run(&mut world, 1.0);
    assert!(height(&world, b) < -1.0);
}

#[test]
fn raycasts_skip_filtered_and_excluded_bodies() {
    let mut world = World::default();
    let ground = floor(&mut world);
    let player = ball(
        &mut world,
        Vec3::new(0.0, 2.0, 0.0),
        CollisionFilter::new(PLAYER, u32::MAX),
    );
    let ghost = ball(
        &mut world,
        Vec3::new(0.0, 4.0, 0.0),
        CollisionFilter::new(GHOST, u32::MAX),
    );
    let down = |filter| {
        world
            .raycast_filtered(Vec3::new(0.0, 10.0, 0.0), Vec3::NEG_Y, 20.0, filter)
            .map(|hit| hit.body)
    };
    assert_eq!(down(QueryFilter::ALL), Some(ghost));
    assert_eq!(down(QueryFilter::mask(WORLD | PLAYER)), Some(player));
    assert_eq!(
        down(QueryFilter::mask(WORLD | PLAYER).excluding(player)),
        Some(ground)
    );
    assert_eq!(down(QueryFilter::mask(0)), None);
}
