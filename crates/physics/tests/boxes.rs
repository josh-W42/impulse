//! Box contract tests against a floor.

use glam::{Quat, Vec3};
use impulse_physics::{BodyDesc, BodyHandle, Collider, Material, SolverMode, World};

const DT: f32 = 1.0 / 60.0;

fn world_with_floor() -> World {
    let mut world = World::default();
    world
        .add_body(BodyDesc::fixed(Vec3::ZERO).with_collider(Collider::Plane { normal: Vec3::Y }))
        .unwrap();
    world
}

fn drop_box(
    world: &mut World,
    half_extents: Vec3,
    position: Vec3,
    orientation: Quat,
) -> BodyHandle {
    let desc = BodyDesc {
        orientation,
        ..BodyDesc::dynamic(1.0, position)
            .with_collider(Collider::Box { half_extents })
            .with_material(Material {
                restitution: 0.0,
                friction: 0.6,
            })
    };
    world.add_body(desc).unwrap()
}

fn run(world: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as u32 {
        world.step(DT);
    }
}

/// Smallest angle (degrees) between world up and any of the body's ± axes.
fn tilt_degrees(orientation: Quat) -> f32 {
    [Vec3::X, Vec3::Y, Vec3::Z]
        .map(|axis| (orientation * axis).dot(Vec3::Y).abs())
        .into_iter()
        .fold(0.0, f32::max)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

#[test]
fn box_dropped_flat_comes_to_rest_flat() {
    let mut world = world_with_floor();
    let h = drop_box(
        &mut world,
        Vec3::new(0.5, 0.3, 0.4),
        Vec3::new(0.0, 2.0, 0.0),
        Quat::IDENTITY,
    );
    run(&mut world, 4.0);
    let b = world.body(h).unwrap();
    assert!(
        tilt_degrees(b.orientation()) < 1.0,
        "tilted {}°",
        tilt_degrees(b.orientation())
    );
    assert!(
        b.velocity().length() < 0.05 && b.angular_velocity().length() < 0.05,
        "still moving"
    );
    assert!(
        (b.position().y - 0.3).abs() < 0.02,
        "rests at {}",
        b.position().y
    );
}

#[test]
fn tilted_box_topples_and_settles_on_a_face() {
    let mut world = world_with_floor();
    let tilt = Quat::from_rotation_z(0.6) * Quat::from_rotation_x(0.3);
    let h = drop_box(
        &mut world,
        Vec3::new(0.25, 0.8, 0.4),
        Vec3::new(0.0, 1.5, 0.0),
        tilt,
    );
    run(&mut world, 6.0);
    let b = world.body(h).unwrap();
    assert!(
        tilt_degrees(b.orientation()) < 5.0,
        "not on a face: {}°",
        tilt_degrees(b.orientation())
    );
    assert!(
        b.velocity().length() < 0.1 && b.angular_velocity().length() < 0.1,
        "still moving"
    );
    assert!(b.position().y > 0.2, "sank: {}", b.position().y);
}

fn world_with_block() -> World {
    let mut world = world_with_floor();
    world
        .add_body(
            BodyDesc::fixed(Vec3::new(0.0, 0.5, 0.0)).with_collider(Collider::Box {
                half_extents: Vec3::splat(0.5),
            }),
        )
        .unwrap();
    world
}

fn add_ball(world: &mut World, position: Vec3, velocity: Vec3) -> BodyHandle {
    world
        .add_body(
            BodyDesc::dynamic(1.0, position)
                .with_collider(Collider::Sphere { radius: 0.3 })
                .with_velocity(velocity),
        )
        .unwrap()
}

#[test]
fn ball_dropped_on_a_box_rests_on_its_top_face() {
    let mut world = world_with_block();
    let ball = add_ball(&mut world, Vec3::new(0.2, 2.0, 0.0), Vec3::ZERO);
    run(&mut world, 3.0);
    let p = world.body(ball).unwrap().position();
    assert!((p.y - 1.3).abs() < 0.02 && (p.x - 0.2).abs() < 0.01, "{p}");
}

#[test]
fn ball_rolls_off_the_edge_of_a_box() {
    let mut world = world_with_block();
    let ball = add_ball(
        &mut world,
        Vec3::new(0.0, 1.3, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
    );
    run(&mut world, 3.0);
    let p = world.body(ball).unwrap().position();
    assert!(
        p.is_finite() && p.x > 0.8 && (p.y - 0.3).abs() < 0.05,
        "should end on the floor: {p}"
    );
}

#[test]
fn box_dropped_on_a_box_comes_to_rest_on_top() {
    let mut world = world_with_floor();
    let cube = Vec3::splat(0.5);
    let bottom = drop_box(&mut world, cube, Vec3::new(0.0, 0.5, 0.0), Quat::IDENTITY);
    let top = drop_box(
        &mut world,
        cube,
        Vec3::new(0.1, 2.0, 0.0),
        Quat::from_rotation_y(0.3),
    );
    run(&mut world, 5.0);
    let (b, t) = (world.body(bottom).unwrap(), world.body(top).unwrap());
    assert!(
        (b.position().y - 0.5).abs() < 0.03,
        "bottom at {}",
        b.position().y
    );
    assert!(
        (t.position().y - 1.5).abs() < 0.05,
        "top at {} (fell through?)",
        t.position().y
    );
    assert!(
        tilt_degrees(t.orientation()) < 3.0,
        "top tilted {}°",
        tilt_degrees(t.orientation())
    );
}

#[test]
fn pile_of_boxes_never_deeply_interpenetrates() {
    let mut world = world_with_floor();
    for i in 0..12 {
        let x = (i % 3) as f32 * 0.7 - 0.7;
        let z = (i / 3 % 2) as f32 * 0.7 - 0.35;
        let tilt = Quat::from_rotation_x(i as f32 * 0.5) * Quat::from_rotation_z(i as f32 * 0.3);
        drop_box(
            &mut world,
            Vec3::new(0.3, 0.25, 0.35),
            Vec3::new(x, 1.0 + i as f32 * 0.6, z),
            tilt,
        );
    }
    let mut deepest = 0.0f32;
    for _ in 0..(8 * 60) {
        world.step(DT);
        for c in impulse_physics::find_contacts(&world) {
            deepest = deepest.max(c.depth);
        }
    }
    // `deepest` only sees contacts that detection reports, so it can't notice
    // missed box–box contacts. Independent check: 12 boxes on 6 spots must pile up
    // (without box–box collisions they all end at floor level, below 0.35 m).
    let highest = world
        .bodies()
        .map(|(_, b)| b.position().y)
        .fold(f32::MIN, f32::max);
    assert!(
        highest > 0.6,
        "no pile formed (highest center {highest}): boxes passed through each other"
    );
    // Fast impacts are detected once per step, so they arrive deep. Classic
    // Baumgarte removes 20% of it at once; the soft step's spring pushes out
    // more gradually (capped at 3 m/s), so its transient peak is a little deeper.
    let bar = match world.solver_settings().mode {
        SolverMode::Classic => 0.15,
        SolverMode::SoftStep => 0.2,
    };
    assert!(deepest < bar, "bodies sank {deepest} m into each other");
    for (_, b) in world.bodies() {
        assert!(b.position().is_finite() && b.position().y > -0.1);
    }
}
