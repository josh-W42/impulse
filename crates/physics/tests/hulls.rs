//! Convex hull contract tests.

use glam::{Quat, Vec3};
use impulse_physics::{BodyDesc, BodyHandle, Collider, HullId, Material, World, find_contacts};

const DT: f32 = 1.0 / 60.0;

fn cube_points(half: f32) -> Vec<Vec3> {
    (0..8)
        .map(|i| {
            let s = |bit: usize| if i & bit == 0 { -half } else { half };
            Vec3::new(s(1), s(2), s(4))
        })
        .collect()
}

fn world_with_floor() -> World {
    let mut world = World::default();
    world
        .add_body(BodyDesc::fixed(Vec3::ZERO).with_collider(Collider::Plane { normal: Vec3::Y }))
        .unwrap();
    world
}

fn material() -> Material {
    Material {
        restitution: 0.0,
        friction: 0.6,
    }
}

fn drop(world: &mut World, collider: Collider, position: Vec3, orientation: Quat) -> BodyHandle {
    let desc = BodyDesc {
        orientation,
        ..BodyDesc::dynamic(1.0, position)
    }
    .with_collider(collider)
    .with_material(material());
    world.add_body(desc).unwrap()
}

fn run(world: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as u32 {
        world.step(DT);
    }
}

#[test]
fn cube_hull_rests_flat_on_four_points() {
    let mut world = world_with_floor();
    let hull = world.add_hull(&cube_points(0.5)).unwrap();
    let h = drop(
        &mut world,
        Collider::Hull { hull },
        Vec3::new(0.0, 2.0, 0.0),
        Quat::IDENTITY,
    );
    run(&mut world, 3.0);
    let b = world.body(h).unwrap();
    assert!(
        (b.position().y - 0.5).abs() < 0.02,
        "rests at {}",
        b.position().y
    );
    assert_eq!(find_contacts(&world).len(), 4);
}

/// Independent cross-check: the same cube as a hull (polytope clipping) and as a
/// box (box routines) should end up in the same place.
#[test]
fn cube_hull_and_box_behave_alike() {
    let tilt = Quat::from_rotation_z(0.4) * Quat::from_rotation_x(0.2);
    let settle = |collider: fn(&mut World) -> Collider| {
        let mut world = world_with_floor();
        let c = collider(&mut world);
        let h = drop(&mut world, c, Vec3::new(0.0, 1.5, 0.0), tilt);
        run(&mut world, 4.0);
        let b = world.body(h).unwrap();
        (b.position(), b.orientation())
    };
    let (hull_pos, hull_rot) = settle(|w| Collider::Hull {
        hull: w.add_hull(&cube_points(0.5)).unwrap(),
    });
    let (box_pos, box_rot) = settle(|_| Collider::Box {
        half_extents: Vec3::splat(0.5),
    });
    assert!(
        (hull_pos.y - box_pos.y).abs() < 0.02,
        "{hull_pos} vs {box_pos}"
    );
    assert!(Vec3::new(hull_pos.x - box_pos.x, 0.0, hull_pos.z - box_pos.z).length() < 0.15);
    // Both land flat (same up-axis alignment up to which face is down).
    let flat = |q: Quat| {
        [Vec3::X, Vec3::Y, Vec3::Z]
            .map(|a| (q * a).dot(Vec3::Y).abs())
            .into_iter()
            .fold(0.0, f32::max)
    };
    assert!(flat(hull_rot) > 0.999 && flat(box_rot) > 0.999);
}

#[test]
fn tetrahedron_settles_on_a_face() {
    let mut world = world_with_floor();
    let hull = world
        .add_hull(&[
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, -0.5, 0.6),
            Vec3::new(-1.0, -0.5, 0.6),
            Vec3::new(0.0, -0.5, -1.1),
        ])
        .unwrap();
    let h = drop(
        &mut world,
        Collider::Hull { hull },
        Vec3::new(0.0, 2.0, 0.0),
        Quat::from_rotation_x(2.5),
    );
    run(&mut world, 5.0);
    let b = world.body(h).unwrap();
    // Some face points straight down.
    let down = world
        .hull(hull)
        .unwrap()
        .faces()
        .iter()
        .map(|f| (b.orientation() * f.normal).dot(Vec3::NEG_Y))
        .fold(f32::MIN, f32::max);
    assert!(down > 0.999, "not resting on a face ({down})");
    assert!(b.velocity().length() < 0.05);
}

#[test]
fn spheres_capsules_boxes_and_hulls_rest_on_a_hull() {
    for (label, collider, rest_height) in [
        ("sphere", Collider::Sphere { radius: 0.3 }, 1.0 + 0.3),
        (
            "box",
            Collider::Box {
                half_extents: Vec3::splat(0.25),
            },
            1.0 + 0.25,
        ),
        (
            "capsule",
            Collider::Capsule {
                half_height: 0.3,
                radius: 0.2,
            },
            1.0 + 0.2,
        ),
    ] {
        let mut world = world_with_floor();
        let block = world.add_hull(&cube_points(0.5)).unwrap();
        world
            .add_body(
                BodyDesc::fixed(Vec3::new(0.0, 0.5, 0.0))
                    .with_collider(Collider::Hull { hull: block }),
            )
            .unwrap();
        // Capsules lie down (axis along x) so they rest at radius height.
        let orientation = if label == "capsule" {
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)
        } else {
            Quat::IDENTITY
        };
        let h = drop(&mut world, collider, Vec3::new(0.0, 2.0, 0.0), orientation);
        run(&mut world, 3.0);
        let y = world.body(h).unwrap().position().y;
        assert!(
            (y - rest_height).abs() < 0.03,
            "{label} rests at {y}, expected {rest_height}"
        );
    }
}

#[test]
fn hull_stacks_on_a_hull() {
    let mut world = world_with_floor();
    let hull: HullId = world.add_hull(&cube_points(0.5)).unwrap();
    let bottom = drop(
        &mut world,
        Collider::Hull { hull },
        Vec3::new(0.0, 0.5, 0.0),
        Quat::IDENTITY,
    );
    let top = drop(
        &mut world,
        Collider::Hull { hull },
        Vec3::new(0.1, 2.0, 0.0),
        Quat::from_rotation_y(0.4),
    );
    run(&mut world, 4.0);
    assert!((world.body(bottom).unwrap().position().y - 0.5).abs() < 0.03);
    assert!(
        (world.body(top).unwrap().position().y - 1.5).abs() < 0.05,
        "top fell through or off"
    );
}

#[test]
fn raycast_hits_a_hull_face() {
    let mut world = World::new(Vec3::ZERO);
    let hull = world.add_hull(&cube_points(0.5)).unwrap();
    world
        .add_body(BodyDesc::fixed(Vec3::new(0.0, 0.0, -5.0)).with_collider(Collider::Hull { hull }))
        .unwrap();
    let hit = world.raycast(Vec3::ZERO, Vec3::NEG_Z, 100.0).unwrap();
    assert!(
        (hit.t - 4.5).abs() < 1e-4 && hit.normal.abs_diff_eq(Vec3::Z, 1e-4),
        "{hit:?}"
    );
}

#[test]
fn clear_forgets_hulls() {
    let mut world = World::default();
    let hull = world.add_hull(&cube_points(0.5)).unwrap();
    world.clear();
    assert!(world.hull(hull).is_none());
    let stale = BodyDesc::dynamic(1.0, Vec3::ZERO).with_collider(Collider::Hull { hull });
    assert!(world.add_body(stale).is_err());
}
