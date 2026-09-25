//! Physical-contract tests for contacts, through the public API only.

use glam::Vec3;
use impulse_physics::{BodyDesc, BodyHandle, Collider, Material, World};

const DT: f32 = 1.0 / 60.0;
const G: f32 = 9.81;
const SLOP: f32 = 0.01;

fn material(restitution: f32, friction: f32) -> Material {
    Material {
        restitution,
        friction,
    }
}

fn ball(
    world: &mut World,
    mass: f32,
    radius: f32,
    position: Vec3,
    velocity: Vec3,
    m: Material,
) -> BodyHandle {
    world
        .add_body(
            BodyDesc::dynamic(mass, position)
                .with_collider(Collider::Sphere { radius })
                .with_velocity(velocity)
                .with_material(m),
        )
        .unwrap()
}

fn floor(world: &mut World, m: Material) {
    world
        .add_body(
            BodyDesc::fixed(Vec3::ZERO)
                .with_collider(Collider::Plane { normal: Vec3::Y })
                .with_material(m),
        )
        .unwrap();
}

fn run(world: &mut World, seconds: f32, mut each: impl FnMut(&World)) {
    for _ in 0..(seconds / DT).round() as u32 {
        world.step(DT);
        each(world);
    }
}

fn pos(world: &World, h: BodyHandle) -> Vec3 {
    world.body(h).unwrap().position()
}

fn vel(world: &World, h: BodyHandle) -> Vec3 {
    world.body(h).unwrap().velocity()
}

#[test]
fn collisions_conserve_momentum() {
    let mut world = World::new(Vec3::ZERO);
    let m = material(0.4, 0.3);
    let a = ball(
        &mut world,
        1.0,
        0.5,
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(3.0, 0.2, 0.0),
        m,
    );
    let b = ball(
        &mut world,
        3.0,
        0.5,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        m,
    );
    let momentum = |w: &World| vel(w, a) * 1.0 + vel(w, b) * 3.0;
    let before = momentum(&world);
    run(&mut world, 2.0, |_| {});
    assert!(
        momentum(&world).abs_diff_eq(before, 1e-4),
        "{} vs {before}",
        momentum(&world)
    );
    assert!(vel(&world, a).x < 0.0, "the pair actually collided");
}

#[test]
fn equal_masses_swap_velocities_in_elastic_head_on_collision() {
    let mut world = World::new(Vec3::ZERO);
    let m = material(1.0, 0.0);
    let a = ball(
        &mut world,
        1.0,
        0.5,
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        m,
    );
    let b = ball(
        &mut world,
        1.0,
        0.5,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::ZERO,
        m,
    );
    run(&mut world, 1.0, |_| {});
    assert!(
        vel(&world, a).abs_diff_eq(Vec3::ZERO, 0.05),
        "a: {}",
        vel(&world, a)
    );
    assert!(
        vel(&world, b).abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 0.05),
        "b: {}",
        vel(&world, b)
    );
}

#[test]
fn elastic_drop_rebounds_to_nearly_its_height() {
    let mut world = World::default();
    floor(&mut world, material(1.0, 0.0));
    let radius = 0.5;
    let h = ball(
        &mut world,
        1.0,
        radius,
        Vec3::new(0.0, 5.0, 0.0),
        Vec3::ZERO,
        material(1.0, 0.0),
    );
    let (mut bounced, mut peak) = (false, 0.0f32);
    run(&mut world, 3.0, |w| {
        let v = vel(w, h).y;
        bounced |= v > 0.0;
        if bounced {
            peak = peak.max(pos(w, h).y);
        }
    });
    let drop = 5.0 - radius;
    assert!(bounced);
    assert!(
        peak >= radius + 0.9 * drop,
        "peak {peak} below 90% of the drop"
    );
}

#[test]
fn inelastic_drop_comes_to_rest_on_the_floor() {
    let mut world = World::default();
    floor(&mut world, material(0.0, 0.5));
    let h = ball(
        &mut world,
        1.0,
        0.5,
        Vec3::new(0.0, 3.0, 0.0),
        Vec3::ZERO,
        material(0.0, 0.5),
    );
    run(&mut world, 3.0, |_| {});
    assert!(
        vel(&world, h).length() < 0.05,
        "still moving: {}",
        vel(&world, h)
    );
    assert!(
        (pos(&world, h).y - 0.5).abs() < SLOP + 1e-3,
        "rests at {}",
        pos(&world, h).y
    );
}

#[test]
fn resting_ball_neither_sinks_nor_jitters() {
    let mut world = World::default();
    floor(&mut world, material(0.5, 0.5));
    let radius = 0.5;
    let h = ball(
        &mut world,
        1.0,
        radius,
        Vec3::new(0.0, radius, 0.0),
        Vec3::ZERO,
        material(0.5, 0.5),
    );
    let (mut lowest, mut fastest) = (f32::MAX, 0.0f32);
    run(&mut world, 5.0, |w| {
        lowest = lowest.min(pos(w, h).y);
        fastest = fastest.max(vel(w, h).length());
    });
    assert!(lowest >= radius - SLOP - 1e-3, "sank to {lowest}");
    assert!(fastest < 0.2, "jittered at {fastest} m/s");
}

#[test]
/// Only while slipping: once v = ω·r the ball rolls and friction stops acting.
/// Slipping lasts 2·v₀/(7·μ·g) ≈ 0.29 s here, so measure at 0.2 s.
fn slipping_ball_decelerates_at_mu_g() {
    let mut world = World::default();
    let m = material(0.0, 0.5);
    floor(&mut world, m);
    let h = ball(
        &mut world,
        1.0,
        0.5,
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::new(5.0, 0.0, 0.0),
        m,
    );
    run(&mut world, 0.2, |_| {});
    let expected = 5.0 - 0.5 * G * 0.2;
    let vx = vel(&world, h).x;
    assert!(
        (vx - expected).abs() < expected * 0.05,
        "vx {vx}, expected {expected}"
    );
}

fn pit(seed: u32) -> (World, Vec<BodyHandle>) {
    let mut world = World::default();
    let m = material(0.5, 0.4);
    for (point, normal) in [
        (Vec3::ZERO, Vec3::Y),
        (Vec3::new(-3.0, 0.0, 0.0), Vec3::X),
        (Vec3::new(3.0, 0.0, 0.0), Vec3::NEG_X),
        (Vec3::new(0.0, 0.0, -3.0), Vec3::Z),
        (Vec3::new(0.0, 0.0, 3.0), Vec3::NEG_Z),
    ] {
        world
            .add_body(
                BodyDesc::fixed(point)
                    .with_collider(Collider::Plane { normal })
                    .with_material(m),
            )
            .unwrap();
    }
    let mut state = seed;
    let mut rand = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32
    };
    let balls = (0..40)
        .map(|i| {
            let p = Vec3::new(rand() * 4.0 - 2.0, 1.0 + i as f32 * 0.5, rand() * 4.0 - 2.0);
            let v = Vec3::new(rand() * 4.0 - 2.0, 0.0, rand() * 4.0 - 2.0);
            ball(&mut world, 0.5 + rand(), 0.2 + rand() * 0.3, p, v, m)
        })
        .collect();
    (world, balls)
}

#[test]
fn crowded_pit_stays_finite_and_contained() {
    let (mut world, balls) = pit(7);
    run(&mut world, 10.0, |_| {});
    for h in balls {
        let p = pos(&world, h);
        assert!(p.is_finite(), "NaN/inf body");
        assert!(
            p.y > -0.1 && p.x.abs() < 3.1 && p.z.abs() < 3.1,
            "escaped the pit: {p}"
        );
    }
}

#[test]
fn contact_simulation_is_deterministic() {
    let (mut a, balls) = pit(3);
    let (mut b, _) = pit(3);
    run(&mut a, 3.0, |_| {});
    run(&mut b, 3.0, |_| {});
    for h in balls {
        assert_eq!(
            pos(&a, h).to_array().map(f32::to_bits),
            pos(&b, h).to_array().map(f32::to_bits)
        );
    }
}

fn angular_momentum(world: &World, bodies: &[BodyHandle]) -> Vec3 {
    bodies
        .iter()
        .map(|&h| {
            let b = world.body(h).unwrap();
            let mass = 1.0 / b.inverse_mass();
            let spin = b.inverse_inertia_world().inverse() * b.angular_velocity();
            b.position().cross(b.velocity() * mass) + spin
        })
        .sum()
}

#[test]
fn sliding_ball_ends_up_rolling_without_slipping() {
    let mut world = World::default();
    let m = material(0.0, 0.5);
    floor(&mut world, m);
    let radius = 0.5;
    let h = ball(
        &mut world,
        1.0,
        radius,
        Vec3::new(0.0, radius, 0.0),
        Vec3::new(5.0, 0.0, 0.0),
        m,
    );
    run(&mut world, 2.0, |_| {});
    let (v, w) = (vel(&world, h), world.body(h).unwrap().angular_velocity());
    // Friction trades linear for angular momentum until v = ω·r: v = 5/7·v₀.
    let expected = 5.0 / 7.0 * 5.0;
    assert!(
        (v.x - expected).abs() < expected * 0.03,
        "v {} vs {expected}",
        v.x
    );
    assert!(
        (v.x + w.z * radius).abs() < 0.05,
        "still slipping: v {} ω {}",
        v.x,
        w.z
    );
}

#[test]
fn glancing_collision_conserves_angular_momentum_and_induces_spin() {
    let mut world = World::new(Vec3::ZERO);
    let m = material(0.0, 0.8);
    // Slow enough that penetration stays within the slop, so only impulses act
    // (positional correction moves bodies and would perturb r × p).
    let a = ball(
        &mut world,
        1.0,
        0.5,
        Vec3::new(-0.6, 0.3, 0.0),
        Vec3::new(0.25, 0.0, 0.0),
        m,
    );
    let b = ball(
        &mut world,
        2.0,
        0.5,
        Vec3::new(0.6, -0.3, 0.0),
        Vec3::new(-0.25, 0.0, 0.0),
        m,
    );
    let before = angular_momentum(&world, &[a, b]);
    run(&mut world, 3.0, |_| {});
    let after = angular_momentum(&world, &[a, b]);
    assert!(
        after.abs_diff_eq(before, 1e-4 * before.length().max(1.0)),
        "{after} vs {before}"
    );
    assert!(
        world.body(a).unwrap().angular_velocity().length() > 0.01,
        "friction should spin a"
    );
}
