//! Joint contract tests: anchors hold, pendulums keep time, limits hold, and
//! joined bodies behave as one island.

use std::f32::consts::{PI, TAU};

use glam::{Quat, Vec3};
use impulse_physics::{
    BodyDesc, BodyHandle, Collider, JointDesc, JointError, SolverSettings, World,
};

const DT: f32 = 1.0 / 60.0;
const G: f32 = 9.81;

fn anchor(world: &mut World, at: Vec3) -> BodyHandle {
    world.add_body(BodyDesc::fixed(at)).unwrap()
}

fn ball(world: &mut World, at: Vec3, radius: f32) -> BodyHandle {
    world
        .add_body(BodyDesc::dynamic(1.0, at).with_collider(Collider::Sphere { radius }))
        .unwrap()
}

fn plank(world: &mut World, at: Vec3, half_extents: Vec3) -> BodyHandle {
    world
        .add_body(BodyDesc::dynamic(1.0, at).with_collider(Collider::Box { half_extents }))
        .unwrap()
}

fn no_sleep(world: &mut World) {
    world.set_solver_settings(SolverSettings {
        sleeping: false,
        ..world.solver_settings()
    });
}

/// Rotation of `q` about +Z (for bodies that only turn about Z).
fn z_angle(q: Quat) -> f32 {
    2.0 * q.z.atan2(q.w)
}

/// Mean time between downward zero crossings of `x`, over `seconds`.
fn period(world: &mut World, bob: BodyHandle, seconds: f32) -> f32 {
    let mut crossings = Vec::new();
    let mut last = world.body(bob).unwrap().position().x;
    for i in 1..=(seconds / DT) as u32 {
        world.step(DT);
        let x = world.body(bob).unwrap().position().x;
        if last > 0.0 && x <= 0.0 {
            crossings.push((i as f32 - x / (x - last)) * DT);
        }
        last = x;
    }
    assert!(crossings.len() >= 3, "pendulum stopped swinging");
    (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f32
}

#[test]
fn pendulums_keep_small_angle_time() {
    let (length, radius, start) = (2.0, 0.1, 0.1_f32);
    let top = Vec3::new(0.0, 5.0, 0.0);
    let bob_at = top + Vec3::new(start.sin(), -start.cos(), 0.0) * length;

    // Distance joint to the bob's center: a point mass on a massless rod.
    let mut world = World::default();
    no_sleep(&mut world);
    let a = anchor(&mut world, top);
    let bob = ball(&mut world, bob_at, radius);
    world
        .add_joint(JointDesc::distance(a, bob, top, bob_at))
        .unwrap();
    let expected = TAU * (length / G).sqrt();
    let measured = period(&mut world, bob, 15.0);
    assert!(
        (measured / expected - 1.0).abs() < 0.03,
        "distance pendulum period {measured}, expected {expected}"
    );

    // Ball-socket at the pivot: a rigid pendulum (sphere inertia about the pivot).
    let mut world = World::default();
    no_sleep(&mut world);
    let a = anchor(&mut world, top);
    let bob = ball(&mut world, bob_at, radius);
    world
        .add_joint(JointDesc::ball_socket(a, bob, top))
        .unwrap();
    let expected = TAU * ((length * length + 0.4 * radius * radius) / (G * length)).sqrt();
    let measured = period(&mut world, bob, 15.0);
    assert!(
        (measured / expected - 1.0).abs() < 0.03,
        "ball-socket pendulum period {measured}, expected {expected}"
    );
}

#[test]
fn ball_socket_anchors_stay_together_while_swinging() {
    // A double pendulum released 45° from vertical.
    let mut world = World::default();
    let top = Vec3::new(0.0, 5.0, 0.0);
    let dir = Vec3::new(1.0, -1.0, 0.0).normalize();
    let tilt = Quat::from_rotation_z(PI / 4.0);
    let half = Vec3::new(0.1, 0.5, 0.1);
    let a = anchor(&mut world, top);
    let links: Vec<BodyHandle> = (0..2)
        .map(|i| {
            let desc = BodyDesc {
                orientation: tilt,
                ..BodyDesc::dynamic(1.0, top + dir * (0.5 + i as f32))
            };
            world
                .add_body(desc.with_collider(Collider::Box { half_extents: half }))
                .unwrap()
        })
        .collect();
    let j0 = world
        .add_joint(JointDesc::ball_socket(a, links[0], top))
        .unwrap();
    let j1 = world
        .add_joint(JointDesc::ball_socket(links[0], links[1], top + dir))
        .unwrap();
    let mut worst: f32 = 0.0;
    for _ in 0..(4.0 / DT) as u32 {
        world.step(DT);
        for j in [j0, j1] {
            let (pa, pb) = world.joint_anchors(j).unwrap();
            worst = worst.max(pa.distance(pb));
        }
    }
    assert!(worst < 0.01, "anchors drifted {worst} m apart");
}

#[test]
fn distance_joint_holds_its_length() {
    let mut world = World::default();
    let top = Vec3::new(0.0, 5.0, 0.0);
    let a = anchor(&mut world, top);
    let bob = world
        .add_body(
            BodyDesc::dynamic(1.0, top - Vec3::Y)
                .with_collider(Collider::Sphere { radius: 0.1 })
                .with_velocity(Vec3::new(3.0, 0.0, 1.0)),
        )
        .unwrap();
    let j = world
        .add_joint(JointDesc::distance(a, bob, top, top - Vec3::Y))
        .unwrap();
    let mut worst: f32 = 0.0;
    for _ in 0..(3.0 / DT) as u32 {
        world.step(DT);
        let (pa, pb) = world.joint_anchors(j).unwrap();
        worst = worst.max((pa.distance(pb) - 1.0).abs());
    }
    assert!(worst < 0.01, "length off by {worst} m");
}

#[test]
fn hinge_turns_only_about_its_axis() {
    let mut world = World::default();
    let pivot = Vec3::new(0.0, 5.0, 0.0);
    let a = anchor(&mut world, pivot);
    let door = world
        .add_body(
            BodyDesc::dynamic(1.0, pivot - Vec3::Y * 0.5)
                .with_collider(Collider::Box {
                    half_extents: Vec3::new(0.3, 0.5, 0.05),
                })
                .with_angular_velocity(Vec3::new(3.0, 2.0, 5.0)),
        )
        .unwrap();
    let j = world
        .add_joint(JointDesc::hinge(a, door, pivot, Vec3::Z))
        .unwrap();
    for _ in 0..(3.0 / DT) as u32 {
        world.step(DT);
        let body = world.body(door).unwrap();
        let axis = body.orientation() * Vec3::Z;
        assert!(axis.dot(Vec3::Z) > 0.999, "hinge axis tilted to {axis}");
        let (pa, pb) = world.joint_anchors(j).unwrap();
        assert!(pa.distance(pb) < 0.01);
    }
    let spin = world.body(door).unwrap().angular_velocity();
    assert!(
        Vec3::new(spin.x, spin.y, 0.0).length() < 0.05,
        "off-axis spin {spin}"
    );
}

#[test]
fn hinge_limits_hold() {
    let mut world = World::default();
    let pivot = Vec3::new(0.0, 5.0, 0.0);
    let a = anchor(&mut world, pivot);
    let door = plank(
        &mut world,
        pivot - Vec3::Y * 0.5,
        Vec3::new(0.05, 0.5, 0.05),
    );
    world.apply_impulse_at(door, Vec3::X * 6.0, pivot - Vec3::Y);
    world
        .add_joint(JointDesc::hinge(a, door, pivot, Vec3::Z).with_limits(-0.5, 0.5))
        .unwrap();
    let (mut low, mut high) = (0.0_f32, 0.0_f32);
    for _ in 0..(4.0 / DT) as u32 {
        world.step(DT);
        let angle = z_angle(world.body(door).unwrap().orientation());
        low = low.min(angle);
        high = high.max(angle);
    }
    assert!(low > -0.55 && high < 0.55, "left limits: [{low}, {high}]");
    assert!(
        high > 0.45,
        "never reached the limit ({high}); test is vacuous"
    );
}

#[test]
fn cone_limit_holds() {
    let mut world = World::default();
    let pivot = Vec3::new(0.0, 5.0, 0.0);
    let a = anchor(&mut world, pivot);
    let arm = plank(
        &mut world,
        pivot - Vec3::Y * 0.5,
        Vec3::new(0.05, 0.5, 0.05),
    );
    world.apply_impulse_at(arm, Vec3::new(5.0, 0.0, 3.0), pivot - Vec3::Y);
    world
        .add_joint(JointDesc::ball_socket(a, arm, pivot).with_cone(Vec3::NEG_Y, 0.5))
        .unwrap();
    let mut widest: f32 = 0.0;
    for _ in 0..(4.0 / DT) as u32 {
        world.step(DT);
        let down = world.body(arm).unwrap().orientation() * Vec3::NEG_Y;
        widest = widest.max(down.angle_between(Vec3::NEG_Y));
    }
    assert!(widest < 0.55, "swung {widest} rad past a 0.5 rad cone");
    assert!(
        widest > 0.45,
        "never reached the cone ({widest}); test is vacuous"
    );
}

#[test]
fn hanging_chain_barely_stretches() {
    let mut world = World::default();
    let top = Vec3::new(0.0, 10.0, 0.0);
    let a = anchor(&mut world, top);
    let half = Vec3::new(0.05, 0.25, 0.05);
    let mut previous = a;
    let mut last = a;
    for i in 0..10 {
        let link = plank(&mut world, top - Vec3::Y * (0.25 + 0.5 * i as f32), half);
        world
            .add_joint(JointDesc::ball_socket(
                previous,
                link,
                top - Vec3::Y * (0.5 * i as f32),
            ))
            .unwrap();
        previous = link;
        last = link;
    }
    for _ in 0..(3.0 / DT) as u32 {
        world.step(DT);
    }
    let body = world.body(last).unwrap();
    let end = body.position() + body.orientation() * (Vec3::NEG_Y * 0.25);
    let stretch = (top.distance(end) - 5.0) / 5.0;
    assert!(stretch < 0.02, "chain stretched {:.1}%", stretch * 100.0);
}

#[test]
fn overlapping_joined_bodies_ignore_each_other_unless_asked() {
    let run = |collide: bool| {
        let mut world = World::new(Vec3::ZERO);
        let half = Vec3::splat(0.5);
        let x = plank(&mut world, Vec3::ZERO, half);
        let y = plank(&mut world, Vec3::new(0.3, 0.0, 0.0), half);
        world
            .add_joint(
                JointDesc::ball_socket(x, y, Vec3::new(0.15, 0.0, 0.0))
                    .with_collide_connected(collide),
            )
            .unwrap();
        for _ in 0..60 {
            world.step(DT);
        }
        world.body(x).unwrap().angular_velocity().length()
            + world.body(y).unwrap().angular_velocity().length()
    };
    assert!(run(false) < 0.01, "joined bodies fought each other");
    assert!(
        run(true) > 0.1,
        "control: colliding joined bodies should push"
    );
}

#[test]
fn chain_sleeps_and_wakes_as_one() {
    let mut world = World::default();
    let top = Vec3::new(0.0, 5.0, 0.0);
    let a = anchor(&mut world, top);
    let half = Vec3::new(0.05, 0.25, 0.05);
    let mut links = Vec::new();
    let mut previous = a;
    for i in 0..3 {
        let link = plank(&mut world, top - Vec3::Y * (0.25 + 0.5 * i as f32), half);
        world
            .add_joint(JointDesc::ball_socket(
                previous,
                link,
                top - Vec3::Y * (0.5 * i as f32),
            ))
            .unwrap();
        links.push(link);
        previous = link;
    }
    for _ in 0..(3.0 / DT) as u32 {
        world.step(DT);
    }
    assert!(links.iter().all(|&l| world.body(l).unwrap().is_asleep()));
    world.apply_impulse_at(links[2], Vec3::X, top - Vec3::Y * 1.5);
    world.step(DT);
    assert!(
        links.iter().all(|&l| !world.body(l).unwrap().is_asleep()),
        "a nudge to the bottom link should wake the whole chain"
    );
}

#[test]
fn removed_bodies_leave_inert_joints() {
    let mut world = World::default();
    let a = ball(&mut world, Vec3::ZERO, 0.1);
    let b = ball(&mut world, Vec3::X, 0.1);
    let j = world
        .add_joint(JointDesc::distance(a, b, Vec3::ZERO, Vec3::X))
        .unwrap();
    world.remove_body(a);
    for _ in 0..60 {
        world.step(DT);
    }
    assert!(world.joint_anchors(j).is_none());
    // b fell freely: nothing held it.
    assert!((world.body(b).unwrap().velocity().y + G).abs() < 1e-3);
}

#[test]
fn rejects_invalid_joints() {
    let mut world = World::default();
    let s = anchor(&mut world, Vec3::ZERO);
    let s2 = anchor(&mut world, Vec3::X);
    let d = ball(&mut world, Vec3::Y, 0.1);
    let gone = ball(&mut world, Vec3::Z, 0.1);
    world.remove_body(gone);
    let p = Vec3::ZERO;
    let cases = [
        (JointDesc::ball_socket(d, d, p), JointError::SameBody),
        (JointDesc::ball_socket(s, gone, p), JointError::UnknownBody),
        (JointDesc::ball_socket(s, s2, p), JointError::NoDynamicBody),
        (
            JointDesc::ball_socket(s, d, Vec3::NAN),
            JointError::NonFinite,
        ),
        (JointDesc::hinge(s, d, p, Vec3::ZERO), JointError::ZeroAxis),
        (
            JointDesc::hinge(s, d, p, Vec3::Z).with_limits(0.5, -0.5),
            JointError::InvalidLimits,
        ),
        (
            JointDesc::hinge(s, d, p, Vec3::Z).with_limits(-4.0, 0.0),
            JointError::InvalidLimits,
        ),
        (
            JointDesc::ball_socket(s, d, p).with_cone(Vec3::Y, 0.0),
            JointError::InvalidCone,
        ),
        (
            JointDesc::ball_socket(s, d, p).with_cone(Vec3::Y, PI),
            JointError::InvalidCone,
        ),
        (
            JointDesc::distance(s, d, Vec3::Y, Vec3::Y),
            JointError::CoincidentAnchors,
        ),
    ];
    for (desc, expected) in cases {
        assert_eq!(world.add_joint(desc), Err(expected), "{desc:?}");
    }
    assert_eq!(world.joints().count(), 0);
}

#[test]
fn clear_and_remove_forget_joints() {
    let mut world = World::default();
    let s = anchor(&mut world, Vec3::ZERO);
    let d = ball(&mut world, Vec3::NEG_Y, 0.1);
    let j = world
        .add_joint(JointDesc::ball_socket(s, d, Vec3::ZERO))
        .unwrap();
    assert!(world.remove_joint(j));
    assert!(!world.remove_joint(j));
    assert!(world.joint_anchors(j).is_none());
    world
        .add_joint(JointDesc::ball_socket(s, d, Vec3::ZERO))
        .unwrap();
    world.clear();
    assert_eq!(world.joints().count(), 0);
}
