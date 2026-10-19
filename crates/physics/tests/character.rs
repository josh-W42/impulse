//! Character controller: walking, walls, slopes, steps, falling and jumping.

use glam::{Quat, Vec3};
use impulse_physics::{
    BodyDesc, BodyHandle, CharacterController, CharacterDesc, Collider, CollisionFilter, World,
};

const DT: f32 = 1.0 / 60.0;

fn world_with_floor() -> World {
    let mut world = World::default();
    world
        .add_body(BodyDesc::fixed(Vec3::ZERO).with_collider(Collider::Plane { normal: Vec3::Y }))
        .unwrap();
    world
}

fn solid(world: &mut World, center: Vec3, half: Vec3, orientation: Quat) -> BodyHandle {
    world
        .add_body(
            BodyDesc {
                orientation,
                ..BodyDesc::fixed(center)
            }
            .with_collider(Collider::Box { half_extents: half }),
        )
        .unwrap()
}

fn character(world: &mut World, feet: Vec3) -> CharacterController {
    CharacterController::new(world, feet, CharacterDesc::default()).unwrap()
}

/// Walks at `velocity` for `seconds`, stepping the world after each update.
fn walk(world: &mut World, c: &mut CharacterController, velocity: Vec3, seconds: f32) {
    for _ in 0..(seconds / DT).round() as u32 {
        c.update(world, velocity, false, DT);
        world.step(DT);
    }
}

/// The body follows the plan exactly.
fn assert_body_at_feet(world: &World, c: &CharacterController) {
    let center = world.body(c.body()).unwrap().position();
    let expected = c.feet() + Vec3::Y * (c.desc().radius + c.desc().half_height);
    assert!(
        center.abs_diff_eq(expected, 1e-4),
        "body {center} vs planned {expected}"
    );
}

#[test]
fn falls_and_lands_on_the_floor() {
    let mut world = world_with_floor();
    let mut c = character(&mut world, Vec3::new(0.0, 3.0, 0.0));
    walk(&mut world, &mut c, Vec3::ZERO, 0.4);
    assert!(!c.is_grounded());
    assert!(c.feet().y < 3.0 && c.feet().y > 0.0);
    walk(&mut world, &mut c, Vec3::ZERO, 1.5);
    assert!(c.is_grounded());
    assert!(
        (c.feet().y - c.desc().skin).abs() < 1e-3,
        "feet at {}",
        c.feet().y
    );
    assert_eq!(c.vertical_speed(), 0.0);
    assert_body_at_feet(&world, &c);
}

#[test]
fn walks_at_the_requested_speed() {
    let mut world = world_with_floor();
    let mut c = character(&mut world, Vec3::new(0.0, 0.5, 0.0));
    walk(&mut world, &mut c, Vec3::ZERO, 1.0);
    let start = c.feet();
    walk(&mut world, &mut c, Vec3::new(3.0, 7.0, -4.0), 1.0); // Y is ignored
    let moved = c.feet() - start;
    assert!(
        moved.abs_diff_eq(Vec3::new(3.0, 0.0, -4.0), 1e-3),
        "moved {moved}"
    );
    assert!(c.is_grounded());
    assert_body_at_feet(&world, &c);
}

#[test]
fn stops_at_a_wall_and_slides_along_it() {
    let mut world = world_with_floor();
    solid(
        &mut world,
        Vec3::new(3.0, 1.5, 0.0),
        Vec3::new(0.5, 1.5, 10.0),
        Quat::IDENTITY,
    );
    let mut c = character(&mut world, Vec3::new(0.0, 0.02, 0.0));
    walk(&mut world, &mut c, Vec3::new(4.0, 0.0, 0.0), 2.0);
    // The wall face is at x = 2.5; the capsule stops a radius and a skin short.
    let stop = 2.5 - c.desc().radius - c.desc().skin;
    assert!(
        (c.feet().x - stop).abs() < 2e-3,
        "stopped at {}",
        c.feet().x
    );
    // Diagonally into the wall: blocked in x, keeps the z part.
    let start = c.feet();
    walk(&mut world, &mut c, Vec3::new(3.0, 0.0, 3.0), 1.0);
    let moved = c.feet() - start;
    assert!(moved.x.abs() < 2e-3, "{moved}");
    assert!((moved.z - 3.0).abs() < 1e-2, "{moved}");
    assert!(c.is_grounded());
}

#[test]
fn stops_in_an_inside_corner_without_jitter() {
    let mut world = world_with_floor();
    solid(
        &mut world,
        Vec3::new(3.0, 1.5, 0.0),
        Vec3::new(0.5, 1.5, 10.0),
        Quat::IDENTITY,
    );
    solid(
        &mut world,
        Vec3::new(0.0, 1.5, -3.0),
        Vec3::new(10.0, 1.5, 0.5),
        Quat::IDENTITY,
    );
    let mut c = character(&mut world, Vec3::new(0.0, 0.02, 0.0));
    walk(&mut world, &mut c, Vec3::new(3.0, 0.0, -3.0), 2.0);
    let settled = c.feet();
    walk(&mut world, &mut c, Vec3::new(3.0, 0.0, -3.0), 1.0);
    assert!(
        c.feet().abs_diff_eq(settled, 1e-4),
        "drifted from {settled} to {}",
        c.feet()
    );
    let stop = 2.5 - c.desc().radius - c.desc().skin;
    assert!(
        (settled.x - stop).abs() < 2e-3 && (settled.z + stop).abs() < 2e-3,
        "{settled}"
    );
}

/// A ramp rising toward +x at `angle`, its top surface through the origin.
fn ramp(world: &mut World, angle: f32) {
    let half = Vec3::new(10.0, 0.5, 5.0);
    let orientation = Quat::from_rotation_z(angle);
    // Shift down along the ramp's own up axis so the top face passes through x = 0, y = 0.
    solid(
        world,
        orientation * Vec3::new(0.0, -half.y, 0.0),
        half,
        orientation,
    );
}

#[test]
fn climbs_walkable_slopes() {
    let mut world = world_with_floor();
    let angle = 30f32.to_radians();
    ramp(&mut world, angle);
    let mut c = character(&mut world, Vec3::new(-3.0, 0.02, 0.0));
    walk(&mut world, &mut c, Vec3::new(2.0, 0.0, 0.0), 3.0);
    let feet = c.feet();
    assert!(feet.x > 2.0, "didn't get up the ramp: {feet}");
    // Standing on the slope: feet (the capsule's lowest point) near the surface.
    assert!((feet.y - feet.x * angle.tan()).abs() < 0.2, "{feet}");
    assert!(c.is_grounded());
    assert!((c.ground_normal().y - angle.cos()).abs() < 1e-2);
}

#[test]
fn cannot_climb_steep_slopes() {
    let mut world = world_with_floor();
    ramp(&mut world, 60f32.to_radians());
    let mut c = character(&mut world, Vec3::new(-3.0, 0.02, 0.0));
    walk(&mut world, &mut c, Vec3::new(2.0, 0.0, 0.0), 3.0);
    assert!(c.feet().y < 0.5, "climbed a 60° slope to {}", c.feet());
    assert!(c.feet().x < 0.5);
}

#[test]
fn walks_down_slopes_without_leaving_the_ground() {
    let mut world = world_with_floor();
    ramp(&mut world, 30f32.to_radians());
    let mut c = character(&mut world, Vec3::new(4.0, 2.4, 0.0));
    walk(&mut world, &mut c, Vec3::ZERO, 1.0);
    assert!(c.is_grounded());
    for _ in 0..60 {
        c.update(&mut world, Vec3::new(-3.0, 0.0, 0.0), false, DT);
        world.step(DT);
        assert!(c.is_grounded(), "left the ground at {}", c.feet());
    }
}

#[test]
fn steps_up_low_ledges_but_not_high_ones() {
    for (height, climbs) in [(0.3, true), (0.6, false)] {
        let mut world = world_with_floor();
        solid(
            &mut world,
            Vec3::new(3.0, height / 2.0, 0.0),
            Vec3::new(1.0, height / 2.0, 3.0),
            Quat::IDENTITY,
        );
        let mut c = character(&mut world, Vec3::new(0.0, 0.02, 0.0));
        walk(&mut world, &mut c, Vec3::new(2.0, 0.0, 0.0), 2.0);
        let on_top = c.feet().y > height - 0.01 && c.feet().x > 2.0;
        assert_eq!(on_top, climbs, "{height} m ledge: feet at {}", c.feet());
        assert!(c.is_grounded());
    }
}

#[test]
fn jumps_and_lands() {
    let mut world = world_with_floor();
    let mut c = character(&mut world, Vec3::new(0.0, 0.02, 0.0));
    walk(&mut world, &mut c, Vec3::ZERO, 0.5);
    c.update(&mut world, Vec3::ZERO, true, DT);
    world.step(DT);
    assert!(!c.is_grounded());
    let mut peak: f32 = 0.0;
    for _ in 0..120 {
        c.update(&mut world, Vec3::ZERO, true, DT); // holding jump doesn't jump again mid-air
        world.step(DT);
        peak = peak.max(c.feet().y);
        if c.is_grounded() {
            break;
        }
    }
    // v²/2g = 25 / 19.62 ≈ 1.27 m.
    assert!((peak - 1.27).abs() < 0.1, "peak {peak}");
    assert!(c.is_grounded());
}

#[test]
fn ceilings_stop_a_jump() {
    let mut world = world_with_floor();
    // Underside at 2.4 m: 0.6 m of headroom above a 1.8 m character.
    solid(
        &mut world,
        Vec3::new(0.0, 2.9, 0.0),
        Vec3::new(3.0, 0.5, 3.0),
        Quat::IDENTITY,
    );
    let mut c = character(&mut world, Vec3::new(0.0, 0.02, 0.0));
    walk(&mut world, &mut c, Vec3::ZERO, 0.5);
    c.update(&mut world, Vec3::ZERO, true, DT);
    world.step(DT);
    let mut peak: f32 = 0.0;
    for _ in 0..120 {
        c.update(&mut world, Vec3::ZERO, false, DT);
        world.step(DT);
        peak = peak.max(c.feet().y);
    }
    assert!(
        peak < 0.6 + 1e-3,
        "head went through the ceiling: feet peaked at {peak}"
    );
    assert!(peak > 0.5, "jump cut short at {peak}");
    assert!(c.is_grounded());
}

#[test]
fn walks_through_groups_outside_its_mask() {
    let mut world = world_with_floor();
    let ghost_wall = world
        .add_body(
            BodyDesc::fixed(Vec3::new(3.0, 1.5, 0.0))
                .with_collider(Collider::Box {
                    half_extents: Vec3::new(0.5, 1.5, 3.0),
                })
                .with_filter(CollisionFilter::new(2, u32::MAX)),
        )
        .unwrap();
    let mut c = CharacterController::new(
        &mut world,
        Vec3::new(0.0, 0.02, 0.0),
        CharacterDesc {
            query_mask: !2,
            ..CharacterDesc::default()
        },
    )
    .unwrap();
    walk(&mut world, &mut c, Vec3::new(3.0, 0.0, 0.0), 2.0);
    assert!(
        c.feet().x > 5.9,
        "blocked by {ghost_wall:?} at {}",
        c.feet()
    );
}

#[test]
fn pushes_dynamic_bodies_it_does_not_query() {
    // Crates in group 2 aren't walls to the character, so its kinematic capsule
    // shoves them.
    let mut world = world_with_floor();
    let crate_body = world
        .add_body(
            BodyDesc::dynamic(5.0, Vec3::new(2.0, 0.5, 0.0))
                .with_collider(Collider::Box {
                    half_extents: Vec3::splat(0.5),
                })
                .with_filter(CollisionFilter::new(2, u32::MAX)),
        )
        .unwrap();
    let mut c = CharacterController::new(
        &mut world,
        Vec3::new(0.0, 0.02, 0.0),
        CharacterDesc {
            query_mask: !2,
            ..CharacterDesc::default()
        },
    )
    .unwrap();
    walk(&mut world, &mut c, Vec3::new(2.0, 0.0, 0.0), 2.0);
    let crate_x = world.body(crate_body).unwrap().position().x;
    assert!(
        crate_x > c.feet().x + 0.7,
        "crate at {crate_x}, character at {}",
        c.feet().x
    );
}

#[test]
fn stops_at_dynamic_bodies_it_queries() {
    let mut world = world_with_floor();
    let crate_body = world
        .add_body(
            BodyDesc::dynamic(5.0, Vec3::new(2.0, 0.5, 0.0)).with_collider(Collider::Box {
                half_extents: Vec3::splat(0.5),
            }),
        )
        .unwrap();
    let mut c = character(&mut world, Vec3::new(0.0, 0.02, 0.0));
    walk(&mut world, &mut c, Vec3::new(2.0, 0.0, 0.0), 2.0);
    let crate_x = world.body(crate_body).unwrap().position().x;
    assert!((crate_x - 2.0).abs() < 0.01, "crate moved to {crate_x}");
    assert!(
        (c.feet().x - (1.5 - 0.35 - 0.02)).abs() < 5e-3,
        "character at {}",
        c.feet()
    );
}

#[test]
fn teleport_moves_the_character_and_its_body() {
    let mut world = world_with_floor();
    let mut c = character(&mut world, Vec3::new(0.0, 3.0, 0.0));
    walk(&mut world, &mut c, Vec3::ZERO, 0.3);
    assert!(c.vertical_speed() < 0.0);
    assert!(c.teleport(&mut world, Vec3::new(10.0, 1.0, 5.0)));
    assert_eq!(c.vertical_speed(), 0.0);
    assert_eq!(c.feet(), Vec3::new(10.0, 1.0, 5.0));
    assert_body_at_feet(&world, &c);
}

#[test]
fn climbs_stairs() {
    let mut world = world_with_floor();
    // Eight 0.2 m steps, 0.4 m deep, rising toward +x from x = 2.
    for i in 0..8 {
        let top = 0.2 * (i + 1) as f32;
        let x = 2.0 + 0.4 * i as f32;
        solid(
            &mut world,
            Vec3::new(x + 0.2, top / 2.0, 0.0),
            Vec3::new(0.2, top / 2.0, 1.5),
            Quat::IDENTITY,
        );
    }
    // A landing at the top, so walking on doesn't lead back down.
    solid(
        &mut world,
        Vec3::new(8.2, 0.8, 0.0),
        Vec3::new(3.0, 0.8, 1.5),
        Quat::IDENTITY,
    );
    let mut c = character(&mut world, Vec3::new(0.0, 0.02, 0.0));
    walk(&mut world, &mut c, Vec3::new(2.0, 0.0, 0.0), 4.0);
    assert!(c.feet().y > 1.6 - 0.01, "reached {}", c.feet());
    assert!(c.is_grounded());
}

/// Deterministic pseudo-random numbers in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Random walking and jumping through clutter never leaves the capsule inside
/// static geometry.
#[test]
fn never_ends_up_inside_static_geometry() {
    let mut world = world_with_floor();
    let mut rng = Lcg(3);
    for _ in 0..40 {
        let center = Vec3::new(rng.next() * 12.0, 0.0, rng.next() * 12.0);
        let half = Vec3::new(
            0.3 + rng.next().abs() * 1.5,
            0.1 + rng.next().abs() * 1.2,
            0.3 + rng.next().abs() * 1.5,
        );
        let tilt =
            Quat::from_rotation_y(rng.next() * 3.0) * Quat::from_rotation_z(rng.next() * 0.5);
        solid(&mut world, center + Vec3::Y * half.y * 0.8, half, tilt);
    }
    // Walls around the area.
    for (center, half) in [
        (Vec3::new(15.0, 2.0, 0.0), Vec3::new(0.5, 2.0, 15.0)),
        (Vec3::new(-15.0, 2.0, 0.0), Vec3::new(0.5, 2.0, 15.0)),
        (Vec3::new(0.0, 2.0, 15.0), Vec3::new(15.0, 2.0, 0.5)),
        (Vec3::new(0.0, 2.0, -15.0), Vec3::new(15.0, 2.0, 0.5)),
    ] {
        solid(&mut world, center, half, Quat::IDENTITY);
    }
    let mut c = character(&mut world, Vec3::new(0.0, 6.0, 0.0));
    let shrunk = Collider::Capsule {
        half_height: c.desc().half_height,
        radius: c.desc().radius - 1e-3,
    };
    let mut velocity = Vec3::ZERO;
    for frame in 0..6000 {
        if frame % 45 == 0 {
            velocity = Vec3::new(rng.next(), 0.0, rng.next()) * 7.0;
        }
        let jump = rng.next() > 0.97;
        c.update(&mut world, velocity, jump, DT);
        world.step(DT);
        let center = c.feet() + Vec3::Y * (c.desc().radius + c.desc().half_height);
        let inside = world.overlap(
            shrunk,
            center,
            Quat::IDENTITY,
            impulse_physics::QueryFilter::ALL.excluding(c.body()),
        );
        assert!(
            inside.is_empty(),
            "frame {frame}: inside {inside:?} at {}",
            c.feet()
        );
        assert!(
            c.feet().y > -0.01,
            "frame {frame}: below the floor at {}",
            c.feet()
        );
    }
}
