//! Stacking stability: the M4 before/after measurements.

use glam::Vec3;
use impulse_physics::{BodyDesc, BodyHandle, Collider, Material, SolverSettings, World};

const DT: f32 = 1.0 / 60.0;
const HALF: f32 = 0.5;

fn world_with_floor() -> World {
    let mut world = World::default();
    world
        .add_body(BodyDesc::fixed(Vec3::ZERO).with_collider(Collider::Plane { normal: Vec3::Y }))
        .unwrap();
    world
}

fn add_cube(world: &mut World, position: Vec3) -> BodyHandle {
    world
        .add_body(
            BodyDesc::dynamic(1.0, position)
                .with_collider(Collider::Box {
                    half_extents: Vec3::splat(HALF),
                })
                .with_material(Material {
                    restitution: 0.0,
                    friction: 0.6,
                }),
        )
        .unwrap()
}

/// Resting contacts may overlap by up to the solver's slop (5 mm), so a stack
/// settles slightly per layer; that's by design and measured separately from drift.
const SINK_PER_LAYER: f32 = 0.007;

/// Worst horizontal drift and vertical sink from the start pose, worst speed in
/// the final second, and worst tilt.
#[derive(Debug)]
struct Stability {
    horizontal_drift: f32,
    sink: f32,
    late_speed: f32,
    tilt_degrees: f32,
}

fn measure(world: &mut World, boxes: &[BodyHandle], seconds: f32) -> Stability {
    let start: Vec<Vec3> = boxes
        .iter()
        .map(|&h| world.body(h).unwrap().position())
        .collect();
    let steps = (seconds / DT).round() as u32;
    let mut late_speed = 0.0f32;
    for step in 0..steps {
        world.step(DT);
        if step >= steps - 60 {
            for &h in boxes {
                let b = world.body(h).unwrap();
                late_speed =
                    late_speed.max(b.velocity().length() + b.angular_velocity().length() * HALF);
            }
        }
    }
    let (mut horizontal_drift, mut sink, mut tilt) = (0.0f32, 0.0f32, 0.0f32);
    for (&h, s) in boxes.iter().zip(&start) {
        let b = world.body(h).unwrap();
        let d = b.position() - *s;
        horizontal_drift = horizontal_drift.max(Vec3::new(d.x, 0.0, d.z).length());
        sink = sink.max(-d.y);
        let up = (b.orientation() * Vec3::Y).dot(Vec3::Y).clamp(-1.0, 1.0);
        tilt = tilt.max(up.acos().to_degrees());
    }
    Stability {
        horizontal_drift,
        sink,
        late_speed,
        tilt_degrees: tilt,
    }
}

fn assert_stable(label: &str, layers: usize, s: &Stability) {
    println!("{label}: {s:?}");
    assert!(
        s.horizontal_drift < 0.01,
        "{label}: drifted {} m ({s:?})",
        s.horizontal_drift
    );
    assert!(
        s.sink < layers as f32 * SINK_PER_LAYER,
        "{label}: sank {} m ({s:?})",
        s.sink
    );
    assert!(
        s.late_speed < 0.01,
        "{label}: still moving at {} m/s ({s:?})",
        s.late_speed
    );
    assert!(
        s.tilt_degrees < 1.0,
        "{label}: tilted {}° ({s:?})",
        s.tilt_degrees
    );
}

pub fn tower(world: &mut World, height: usize) -> Vec<BodyHandle> {
    (0..height)
        .map(|i| add_cube(world, Vec3::new(0.0, HALF + i as f32 * 2.0 * HALF, 0.0)))
        .collect()
}

pub fn pyramid(world: &mut World, base: usize) -> Vec<BodyHandle> {
    let mut boxes = Vec::new();
    for row in 0..base {
        let count = base - row;
        for i in 0..count {
            let x = (i as f32 - (count - 1) as f32 / 2.0) * 2.05 * HALF;
            boxes.push(add_cube(
                world,
                Vec3::new(x, HALF + row as f32 * 2.0 * HALF, 0.0),
            ));
        }
    }
    boxes
}

#[test]
fn tower_of_five_stands_still() {
    let mut world = world_with_floor();
    let boxes = tower(&mut world, 5);
    let s = measure(&mut world, &boxes, 5.0);
    assert_stable("tower of 5", 5, &s);
}

#[test]
fn pyramid_of_fifteen_stands_still() {
    let mut world = world_with_floor();
    let boxes = pyramid(&mut world, 5);
    let s = measure(&mut world, &boxes, 5.0);
    assert_stable("pyramid of 15", 5, &s);
}

/// Known M4 limit: a 10-high tower stands (M3's solver collapsed it: 12 m drift,
/// 180° tilt, 5.3 m/s) but keeps a residual wobble — Gauss–Seidel doesn't fully
/// converge through 10 layers of redundant 4-point contacts. Bounds are loose on
/// purpose; soft step (deferred) or sleeping (M5) would tighten them.
#[test]
fn tower_of_ten_stands_with_bounded_wobble() {
    let mut world = world_with_floor();
    let boxes = tower(&mut world, 10);
    let s = measure(&mut world, &boxes, 5.0);
    println!("tower of 10: {s:?}");
    assert!(
        s.horizontal_drift < 0.15,
        "drifted {} m ({s:?})",
        s.horizontal_drift
    );
    assert!(
        s.late_speed < 0.2,
        "wobbling at {} m/s ({s:?})",
        s.late_speed
    );
    assert!(s.tilt_degrees < 2.0, "tilted {}° ({s:?})", s.tilt_degrees);
}

/// Evidence that warm starting is what holds stacks, not just a lucky result:
/// same tower, same iterations, only warm starting toggled.
#[test]
fn warm_starting_is_what_holds_the_tower() {
    let run = |warm_starting: bool| {
        let mut world = world_with_floor();
        world.set_solver_settings(SolverSettings {
            warm_starting,
            ..SolverSettings::default()
        });
        let boxes = tower(&mut world, 5);
        measure(&mut world, &boxes, 5.0)
    };
    let (with, without) = (run(true), run(false));
    println!(
        "tower of 5 — warm: {with:?}
              cold: {without:?}"
    );
    assert!(
        with.late_speed < 0.01,
        "warm-started tower moving: {with:?}"
    );
    assert!(
        without.horizontal_drift > 10.0 * with.horizontal_drift.max(1e-3),
        "warm starting made no real difference: {with:?} vs {without:?}"
    );
}

#[test]
fn dropped_box_settles_flat_onto_a_tower() {
    let mut world = world_with_floor();
    let mut boxes = tower(&mut world, 4);
    boxes.push(add_cube(&mut world, Vec3::new(0.05, 6.0, 0.0)));
    // Only a gentle landing: the dropped box must end up resting on top, level.
    for _ in 0..(5 * 60) {
        world.step(DT);
    }
    let top = world.body(*boxes.last().unwrap()).unwrap();
    let up = (top.orientation() * Vec3::Y)
        .dot(Vec3::Y)
        .abs()
        .min(1.0)
        .acos()
        .to_degrees();
    assert!(
        (top.position().y - 4.5).abs() < 0.03,
        "top at {}",
        top.position().y
    );
    assert!(up < 2.0, "top tilted {up}°");
}
