//! Native step-time benchmark: `cargo run --release -p impulse-physics --example bench`.
//!
//! Drops N mixed spheres/boxes into a walled pit and times every step of a 5 s
//! simulation (300 steps at 60 Hz). Deterministic scene, so runs are comparable.

use std::time::Instant;

use glam::{Quat, Vec3};
use impulse_physics::{BodyDesc, Collider, Material, World};

const DT: f32 = 1.0 / 60.0;
const STEPS: usize = 300;

fn spread(i: usize, k: f32) -> f32 {
    (i as f32 * k).fract()
}

/// Walled pit sized so `count` bodies fit a few layers deep, plus the bodies.
pub fn rain(world: &mut World, count: usize) {
    let half = (count as f32).sqrt() * 0.45 + 3.0;
    for (point, normal) in [
        (Vec3::ZERO, Vec3::Y),
        (Vec3::new(-half, 0.0, 0.0), Vec3::X),
        (Vec3::new(half, 0.0, 0.0), Vec3::NEG_X),
        (Vec3::new(0.0, 0.0, -half), Vec3::Z),
        (Vec3::new(0.0, 0.0, half), Vec3::NEG_Z),
    ] {
        world
            .add_body(BodyDesc::fixed(point).with_collider(Collider::Plane { normal }))
            .unwrap();
    }
    let per_layer = ((2.0 * half - 2.0) / 1.2).floor().max(1.0) as usize;
    for i in 0..count {
        let (col, row, layer) = (
            i % per_layer,
            (i / per_layer) % per_layer,
            i / (per_layer * per_layer),
        );
        let position = Vec3::new(
            -half + 1.0 + col as f32 * 1.2,
            1.0 + layer as f32 * 1.2,
            -half + 1.0 + row as f32 * 1.2,
        );
        let size = 0.25 + 0.2 * spread(i, 0.37);
        let collider = if i % 2 == 0 {
            Collider::Sphere { radius: size }
        } else {
            Collider::Box {
                half_extents: Vec3::new(size, size * 0.8, size * 0.9),
            }
        };
        let desc = BodyDesc {
            orientation: Quat::from_rotation_y(spread(i, 0.53) * 6.0),
            ..BodyDesc::dynamic(1.0, position)
        }
        .with_collider(collider)
        .with_material(Material {
            restitution: 0.2,
            friction: 0.5,
        });
        world.add_body(desc).unwrap();
    }
}

fn main() {
    println!("| bodies | mean step (ms) | p95 step (ms) | max step (ms) |");
    println!("|---|---|---|---|");
    for count in [100, 500, 1000] {
        let mut world = World::default();
        rain(&mut world, count);
        let mut times: Vec<f64> = (0..STEPS)
            .map(|_| {
                let start = Instant::now();
                world.step(DT);
                start.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        times.sort_by(f64::total_cmp);
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let p95 = times[(times.len() * 95) / 100];
        println!(
            "| {count} | {mean:.2} | {p95:.2} | {:.2} |",
            times.last().unwrap()
        );
    }
}
