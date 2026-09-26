//! Island sleeping: resting groups stop simulating; touching them wakes them.

use glam::Vec3;
use impulse_physics::{BodyDesc, BodyHandle, Collider, Material, SolverSettings, World};

const DT: f32 = 1.0 / 60.0;

fn world_with_floor() -> World {
    let mut world = World::default();
    world
        .add_body(BodyDesc::fixed(Vec3::ZERO).with_collider(Collider::Plane { normal: Vec3::Y }))
        .unwrap();
    world
}

fn stack(world: &mut World, height: usize) -> Vec<BodyHandle> {
    (0..height)
        .map(|i| {
            world
                .add_body(
                    BodyDesc::dynamic(1.0, Vec3::new(0.0, 0.5 + i as f32, 0.0))
                        .with_collider(Collider::Box {
                            half_extents: Vec3::splat(0.5),
                        })
                        .with_material(Material {
                            restitution: 0.0,
                            friction: 0.6,
                        }),
                )
                .unwrap()
        })
        .collect()
}

fn run(world: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as u32 {
        world.step(DT);
    }
}

fn asleep(world: &World, bodies: &[BodyHandle]) -> bool {
    bodies.iter().all(|&h| world.body(h).unwrap().is_asleep())
}

#[test]
fn resting_stack_falls_asleep_and_stays_put() {
    let mut world = world_with_floor();
    let boxes = stack(&mut world, 3);
    run(&mut world, 2.0);
    assert!(asleep(&world, &boxes), "stack still awake after 2 s");
    let bits = |w: &World| -> Vec<[u32; 3]> {
        boxes
            .iter()
            .map(|&h| w.body(h).unwrap().position().to_array().map(f32::to_bits))
            .collect()
    };
    let before = bits(&world);
    run(&mut world, 2.0);
    assert_eq!(bits(&world), before, "sleeping bodies moved");
}

#[test]
fn a_ball_landing_on_a_sleeping_stack_wakes_it() {
    let mut world = world_with_floor();
    let boxes = stack(&mut world, 3);
    run(&mut world, 2.0);
    assert!(asleep(&world, &boxes));
    world
        .add_body(
            BodyDesc::dynamic(5.0, Vec3::new(0.3, 5.0, 0.0))
                .with_collider(Collider::Sphere { radius: 0.4 }),
        )
        .unwrap();
    let mut woke = false;
    for _ in 0..120 {
        world.step(DT);
        woke |= boxes.iter().all(|&h| !world.body(h).unwrap().is_asleep());
    }
    assert!(woke, "impact never woke the stack");
}

#[test]
fn nothing_sleeps_when_sleeping_is_disabled() {
    let mut world = world_with_floor();
    world.set_solver_settings(SolverSettings {
        sleeping: false,
        ..SolverSettings::default()
    });
    let boxes = stack(&mut world, 3);
    run(&mut world, 3.0);
    assert!(boxes.iter().all(|&h| !world.body(h).unwrap().is_asleep()));
}

#[test]
fn a_falling_body_never_sleeps() {
    let mut world = World::default();
    let ball = world
        .add_body(
            BodyDesc::dynamic(1.0, Vec3::ZERO).with_collider(Collider::Sphere { radius: 0.5 }),
        )
        .unwrap();
    run(&mut world, 3.0);
    assert!(!world.body(ball).unwrap().is_asleep());
}
