//! Soft step (M11) contract tests, measured against the classic solver.
//! Sleeping is off throughout, so resting motion can't hide behind sleep.

use glam::Vec3;
use impulse_physics::{BodyDesc, BodyHandle, Collider, Material, SolverMode, SolverSettings, World};

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
    let start: Vec<Vec3> = boxes.iter().map(|&h| world.body(h).unwrap().position()).collect();
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

#[test]
fn baseline_classic_towers() {
    for height in [5, 10, 20] {
        let s = tower_stability(SolverMode::Classic, height);
        println!("classic tower of {height}: {s:?}");
    }
}
