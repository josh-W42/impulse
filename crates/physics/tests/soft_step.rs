//! Soft step (M11) contract tests, measured against the classic solver.
//! Sleeping is off throughout, so resting motion can't hide behind sleep.

use glam::Vec3;
use impulse_physics::{
    BodyDesc, BodyHandle, Collider, Material, SolverMode, SolverSettings, World,
};

const DT: f32 = 1.0 / 60.0;
const HALF: f32 = 0.5;

fn world(mode: SolverMode) -> World {
    let mut world = World::default();
    world.set_solver_settings(SolverSettings {
        mode,
        sleeping: false,
        ..world.solver_settings()
    });
    world
        .add_body(BodyDesc::fixed(Vec3::ZERO).with_collider(Collider::Plane { normal: Vec3::Y }))
        .unwrap();
    world
}

fn cube(world: &mut World, position: Vec3) -> BodyHandle {
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

fn tower(world: &mut World, height: usize) -> Vec<BodyHandle> {
    (0..height)
        .map(|i| cube(world, Vec3::new(0.0, HALF + i as f32 * 2.0 * HALF, 0.0)))
        .collect()
}

/// Worst horizontal drift from the start, worst speed in the final second
/// (linear + angular at the box's half size), and worst tilt.
#[derive(Debug, Clone, Copy)]
pub struct Stability {
    pub drift: f32,
    pub late_speed: f32,
    pub tilt_degrees: f32,
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
    let (mut drift, mut tilt) = (0.0f32, 0.0f32);
    for (&h, s) in boxes.iter().zip(&start) {
        let b = world.body(h).unwrap();
        let d = b.position() - *s;
        drift = drift.max(Vec3::new(d.x, 0.0, d.z).length());
        let up = (b.orientation() * Vec3::Y).dot(Vec3::Y).clamp(-1.0, 1.0);
        tilt = tilt.max(up.acos().to_degrees());
    }
    Stability {
        drift,
        late_speed,
        tilt_degrees: tilt,
    }
}

fn tower_stability(mode: SolverMode, height: usize) -> Stability {
    let mut w = world(mode);
    let boxes = tower(&mut w, height);
    measure(&mut w, &boxes, 10.0)
}

/// The M11 goal: under the soft step, the 10-high tower meets the bar the
/// 5-high tower meets under classic (and classic misses it by far).
#[test]
fn soft_step_holds_a_tower_of_ten_to_the_five_tower_bar() {
    let soft = tower_stability(SolverMode::SoftStep, 10);
    assert!(soft.drift < 0.01, "drift {soft:?}");
    assert!(soft.late_speed < 0.01, "still moving {soft:?}");
    assert!(soft.tilt_degrees < 1.0, "tilt {soft:?}");
    let classic = tower_stability(SolverMode::Classic, 10);
    assert!(
        classic.late_speed > 10.0 * soft.late_speed,
        "soft should beat classic clearly: {soft:?} vs {classic:?}"
    );
}

/// Two boxes started half inside each other are pushed apart no faster than
/// the soft step's 3 m/s cap, not launched.
#[test]
fn deep_overlaps_separate_gently() {
    let mut w = World::new(Vec3::ZERO);
    w.set_solver_settings(SolverSettings {
        mode: SolverMode::SoftStep,
        sleeping: false,
        ..w.solver_settings()
    });
    let a = cube(&mut w, Vec3::ZERO);
    let b = cube(&mut w, Vec3::new(0.5, 0.0, 0.0));
    let mut fastest = 0.0f32;
    for _ in 0..120 {
        w.step(DT);
        let relative = w.body(b).unwrap().velocity() - w.body(a).unwrap().velocity();
        fastest = fastest.max(relative.length());
    }
    assert!(fastest <= 3.0 + 0.05, "separated at {fastest} m/s");
    let gap = w.body(b).unwrap().position().x - w.body(a).unwrap().position().x;
    assert!(gap > 0.95, "still overlapping: centers {gap} m apart");
}

fn energy(world: &World, bodies: &[BodyHandle]) -> f32 {
    bodies
        .iter()
        .map(|&h| {
            let b = world.body(h).unwrap();
            let m = 1.0 / b.inverse_mass();
            let spin = b.angular_velocity();
            0.5 * m * b.velocity().length_squared()
                + 0.5 * spin.dot(b.inverse_inertia_world().inverse() * spin)
                + m * 9.81 * b.position().y
        })
        .sum()
}

/// A resting pyramid only ever loses energy (soft corrections must not pump).
#[test]
fn a_resting_pyramid_gains_no_energy() {
    let mut w = world(SolverMode::SoftStep);
    let mut boxes = Vec::new();
    for row in 0..5 {
        let count = 5 - row;
        for i in 0..count {
            let x = (i as f32 - (count - 1) as f32 / 2.0) * 2.05 * HALF;
            boxes.push(cube(
                &mut w,
                Vec3::new(x, HALF + row as f32 * 2.0 * HALF, 0.0),
            ));
        }
    }
    for _ in 0..120 {
        w.step(DT);
    }
    let settled = energy(&w, &boxes);
    let mut highest = settled;
    for _ in 0..(8 * 60) {
        w.step(DT);
        highest = highest.max(energy(&w, &boxes));
    }
    assert!(highest - settled < 1e-3, "gained {} J", highest - settled);
}

/// Substepped gravity is exact for velocity, and moves the body at least as
/// accurately as classic's single semi-implicit Euler step.
#[test]
fn free_fall_is_exact_in_velocity_and_no_worse_in_position() {
    let fall = |mode| {
        let mut w = World::default();
        w.set_solver_settings(SolverSettings {
            mode,
            ..w.solver_settings()
        });
        let h = w
            .add_body(BodyDesc::dynamic(1.0, Vec3::new(0.0, 100.0, 0.0)))
            .unwrap();
        for _ in 0..120 {
            w.step(DT);
        }
        let b = w.body(h).unwrap();
        (b.velocity().y, b.position().y)
    };
    let (v, soft_y) = fall(SolverMode::SoftStep);
    let (_, classic_y) = fall(SolverMode::Classic);
    assert!((v + 9.81 * 2.0).abs() < 1e-3, "velocity {v}");
    let exact = 100.0 - 0.5 * 9.81 * 4.0;
    assert!((soft_y - exact).abs() <= (classic_y - exact).abs() + 1e-4);
}

#[test]
fn identical_soft_step_worlds_stay_bit_identical() {
    let run = || {
        let mut w = world(SolverMode::SoftStep);
        let boxes = tower(&mut w, 6);
        w.apply_impulse_at(boxes[5], Vec3::new(2.0, 0.0, 1.0), Vec3::new(0.0, 5.8, 0.0));
        for _ in 0..240 {
            w.step(DT);
        }
        boxes
            .iter()
            .map(|&h| w.body(h).unwrap().position().to_array().map(f32::to_bits))
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}

/// Switching solvers mid-run (the demo's toggle) doesn't jolt a resting stack:
/// the warm-start cache is shared.
#[test]
fn switching_solvers_keeps_a_stack_calm() {
    let mut w = world(SolverMode::Classic);
    let boxes = tower(&mut w, 5);
    for _ in 0..120 {
        w.step(DT);
    }
    w.set_solver_settings(SolverSettings {
        mode: SolverMode::SoftStep,
        ..w.solver_settings()
    });
    let s = measure(&mut w, &boxes, 3.0);
    assert!(
        s.drift < 0.01 && s.late_speed < 0.01 && s.tilt_degrees < 1.0,
        "{s:?}"
    );
}

/// Prints the tower table in docs/plans/2026-09-27-m11-soft-step.md
/// (`cargo test --release -p impulse-physics --test soft_step -- --ignored --nocapture`).
#[test]
#[ignore]
fn report_towers() {
    for height in [5, 10, 20] {
        for mode in [SolverMode::Classic, SolverMode::SoftStep] {
            println!(
                "{mode:?} tower of {height}: {:?}",
                tower_stability(mode, height)
            );
        }
    }
}
