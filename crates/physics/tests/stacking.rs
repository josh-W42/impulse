//! Stacking stability: the M4 before/after measurements.

use glam::Vec3;
use impulse_physics::{BodyDesc, BodyHandle, Collider, Material, World};

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

/// Worst drift from the start pose, worst speed in the final second, and worst tilt.
#[derive(Debug)]
struct Stability {
    drift: f32,
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
    let mut drift = 0.0f32;
    let mut tilt = 0.0f32;
    for (&h, s) in boxes.iter().zip(&start) {
        let b = world.body(h).unwrap();
        drift = drift.max(b.position().distance(*s));
        let up = (b.orientation() * Vec3::Y).dot(Vec3::Y).clamp(-1.0, 1.0);
        tilt = tilt.max(up.acos().to_degrees());
    }
    Stability {
        drift,
        late_speed,
        tilt_degrees: tilt,
    }
}

fn assert_stable(label: &str, s: &Stability) {
    println!("{label}: {s:?}");
    assert!(s.drift < 0.01, "{label}: drifted {} m ({s:?})", s.drift);
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
fn tower_of_ten_stands_still() {
    let mut world = world_with_floor();
    let boxes = tower(&mut world, 10);
    let s = measure(&mut world, &boxes, 5.0);
    assert_stable("tower of 10", &s);
}

#[test]
fn pyramid_of_fifteen_stands_still() {
    let mut world = world_with_floor();
    let boxes = pyramid(&mut world, 5);
    let s = measure(&mut world, &boxes, 5.0);
    assert_stable("pyramid of 15", &s);
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
