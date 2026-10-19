//! Shape casts and overlap queries.

use glam::{Quat, Vec3};
use impulse_physics::{
    BodyDesc, BodyHandle, CAST_TOLERANCE, Collider, CollisionFilter, QueryFilter, World,
};

/// Distances agree with geometry to within this (the cast stops up to
/// `CAST_TOLERANCE` short of the surface).
const EPS: f32 = 2e-3;

fn fixed(world: &mut World, position: Vec3, orientation: Quat, collider: Collider) -> BodyHandle {
    world
        .add_body(
            BodyDesc {
                orientation,
                ..BodyDesc::fixed(position)
            }
            .with_collider(collider),
        )
        .unwrap()
}

fn sphere(radius: f32) -> Collider {
    Collider::Sphere { radius }
}

fn capsule(half_height: f32, radius: f32) -> Collider {
    Collider::Capsule {
        half_height,
        radius,
    }
}

fn cuboid(half: Vec3) -> Collider {
    Collider::Box { half_extents: half }
}

#[test]
fn sphere_cast_onto_a_plane() {
    let mut world = World::default();
    let floor = fixed(
        &mut world,
        Vec3::ZERO,
        Quat::IDENTITY,
        Collider::Plane { normal: Vec3::Y },
    );
    let hit = world
        .shape_cast(
            sphere(0.5),
            Vec3::new(1.0, 5.0, 2.0),
            Quat::IDENTITY,
            Vec3::NEG_Y,
            10.0,
            QueryFilter::ALL,
        )
        .unwrap();
    assert_eq!(hit.body, floor);
    assert!((hit.distance - 4.5).abs() < EPS, "{}", hit.distance);
    assert!(hit.normal.abs_diff_eq(Vec3::Y, 1e-5));
    assert!(
        hit.point.abs_diff_eq(Vec3::new(1.0, 0.0, 2.0), EPS),
        "{}",
        hit.point
    );
}

#[test]
fn capsule_cast_into_a_wall() {
    let mut world = World::default();
    let wall = fixed(
        &mut world,
        Vec3::new(5.0, 1.0, 0.0),
        Quat::IDENTITY,
        cuboid(Vec3::new(0.5, 2.0, 2.0)),
    );
    // Standing capsule (radius 0.35) walking +x: its side touches the wall face at x = 4.5.
    let hit = world
        .shape_cast(
            capsule(0.55, 0.35),
            Vec3::new(0.0, 0.9, 0.3),
            Quat::IDENTITY,
            Vec3::X,
            10.0,
            QueryFilter::ALL,
        )
        .unwrap();
    assert_eq!(hit.body, wall);
    assert!(
        (hit.distance - (4.5 - 0.35)).abs() < EPS,
        "{}",
        hit.distance
    );
    assert!(hit.normal.abs_diff_eq(Vec3::NEG_X, 1e-3), "{}", hit.normal);
    assert!((hit.point.x - 4.5).abs() < EPS);
}

#[test]
fn cast_against_a_rotated_box_corner() {
    let mut world = World::default();
    fixed(
        &mut world,
        Vec3::ZERO,
        Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
        cuboid(Vec3::ONE),
    );
    let hit = world
        .shape_cast(
            sphere(0.5),
            Vec3::new(5.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::NEG_X,
            10.0,
            QueryFilter::ALL,
        )
        .unwrap();
    // The corner sits at x = √2.
    assert!(
        (hit.distance - (5.0 - 2f32.sqrt() - 0.5)).abs() < EPS,
        "{}",
        hit.distance
    );
    assert!(
        hit.point.abs_diff_eq(Vec3::new(2f32.sqrt(), 0.0, 0.0), EPS),
        "{}",
        hit.point
    );
}

#[test]
fn casts_that_miss() {
    let mut world = World::default();
    fixed(
        &mut world,
        Vec3::new(5.0, 0.0, 0.0),
        Quat::IDENTITY,
        cuboid(Vec3::ONE),
    );
    let cast = |from: Vec3, dir: Vec3, max: f32| {
        world.shape_cast(
            sphere(0.5),
            from,
            Quat::IDENTITY,
            dir,
            max,
            QueryFilter::ALL,
        )
    };
    assert!(
        cast(Vec3::ZERO, Vec3::NEG_X, 100.0).is_none(),
        "moving away"
    );
    assert!(
        cast(Vec3::new(0.0, 2.0, 0.0), Vec3::X, 100.0).is_none(),
        "passing above (gap 0.5)"
    );
    assert!(
        cast(Vec3::ZERO, Vec3::X, 3.0).is_none(),
        "stops short (needs 3.5)"
    );
    assert!(cast(Vec3::ZERO, Vec3::X, 3.6).is_some());
    assert!(cast(Vec3::ZERO, Vec3::ZERO, 10.0).is_none(), "no direction");
}

#[test]
fn starting_inside_hits_at_zero_and_points_out() {
    let mut world = World::default();
    fixed(&mut world, Vec3::ZERO, Quat::IDENTITY, cuboid(Vec3::ONE));
    // Overlapping only through the radius (cores apart).
    let shallow = world
        .shape_cast(
            sphere(0.5),
            Vec3::new(1.3, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::X,
            5.0,
            QueryFilter::ALL,
        )
        .unwrap();
    assert_eq!(shallow.distance, 0.0);
    assert!(
        shallow.normal.abs_diff_eq(Vec3::X, 1e-3),
        "{}",
        shallow.normal
    );
    // Center inside the box (cores overlap).
    let deep = world
        .shape_cast(
            sphere(0.5),
            Vec3::new(0.0, 0.8, 0.0),
            Quat::IDENTITY,
            Vec3::X,
            5.0,
            QueryFilter::ALL,
        )
        .unwrap();
    assert_eq!(deep.distance, 0.0);
    assert!(deep.normal.abs_diff_eq(Vec3::Y, 1e-3), "{}", deep.normal);
}

#[test]
fn nearest_body_wins_and_filters_apply() {
    let mut world = World::default();
    let near = fixed(
        &mut world,
        Vec3::new(3.0, 0.0, 0.0),
        Quat::IDENTITY,
        sphere(0.5),
    );
    let far = world
        .add_body(
            BodyDesc::fixed(Vec3::new(6.0, 0.0, 0.0))
                .with_collider(sphere(0.5))
                .with_filter(CollisionFilter::new(2, u32::MAX)),
        )
        .unwrap();
    let cast = |filter| {
        world
            .shape_cast(
                sphere(0.5),
                Vec3::ZERO,
                Quat::IDENTITY,
                Vec3::X,
                20.0,
                filter,
            )
            .map(|h| (h.body, h.distance))
    };
    let (body, distance) = cast(QueryFilter::ALL).unwrap();
    assert_eq!(body, near);
    assert!((distance - 2.0).abs() < EPS);
    assert_eq!(cast(QueryFilter::ALL.excluding(near)).unwrap().0, far);
    assert_eq!(cast(QueryFilter::mask(2)).unwrap().0, far);
    assert!(cast(QueryFilter::mask(4)).is_none());
}

const HULL: [Vec3; 5] = [
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(1.0, -0.5, 0.6),
    Vec3::new(-1.0, -0.5, 0.6),
    Vec3::new(0.0, -0.5, -1.0),
    Vec3::new(0.3, 0.2, 0.9),
];

/// Casts every shape against every target shape and checks the distance
/// against stepping the shape along the path with `overlap`.
#[test]
fn cast_distance_matches_stepped_overlap() {
    for target in 0..4 {
        let mut world = World::default();
        let hull = world.add_hull(&HULL).unwrap();
        let shapes = [
            sphere(0.4),
            capsule(0.5, 0.3),
            cuboid(Vec3::new(0.3, 0.5, 0.4)),
            Collider::Hull { hull },
        ];
        let tilt = Quat::from_rotation_z(0.7) * Quat::from_rotation_x(0.3);
        let body = fixed(&mut world, Vec3::new(4.0, 0.2, -0.1), tilt, shapes[target]);
        for caster in shapes {
            let orientation = Quat::from_rotation_y(0.4);
            let start = Vec3::new(0.0, 0.3, 0.0);
            let dir = Vec3::new(1.0, 0.05, -0.02).normalize();
            let target = shapes[target];
            let hit = world
                .shape_cast(caster, start, orientation, dir, 10.0, QueryFilter::ALL)
                .unwrap_or_else(|| panic!("{caster:?} missed {target:?}"));
            assert_eq!(hit.body, body);
            // First millimetre step along the path where the shapes overlap.
            let overlapping = |t: f32| {
                !world
                    .overlap(caster, start + dir * t, orientation, QueryFilter::ALL)
                    .is_empty()
            };
            let stepped = (0..10_000)
                .map(|i| i as f32 * 1e-3)
                .find(|&t| overlapping(t))
                .unwrap();
            assert!(
                hit.distance <= stepped && hit.distance > stepped - 1e-3 - 2.0 * CAST_TOLERANCE,
                "{caster:?} → {target:?}: cast {} vs stepped {stepped}",
                hit.distance
            );
            assert!(
                !overlapping(hit.distance),
                "{caster:?} → {target:?}: overlapping at the reported distance {}",
                hit.distance
            );
            assert!(
                hit.normal.dot(dir) < 0.0,
                "the normal faces back along the path"
            );
        }
    }
}

#[test]
fn overlap_lists_bodies_in_handle_order() {
    let mut world = World::default();
    let floor = fixed(
        &mut world,
        Vec3::ZERO,
        Quat::IDENTITY,
        Collider::Plane { normal: Vec3::Y },
    );
    let a = fixed(
        &mut world,
        Vec3::new(1.0, 0.5, 0.0),
        Quat::IDENTITY,
        cuboid(Vec3::splat(0.5)),
    );
    let b = fixed(
        &mut world,
        Vec3::new(-1.0, 0.5, 0.0),
        Quat::IDENTITY,
        sphere(0.5),
    );
    let _far = fixed(
        &mut world,
        Vec3::new(10.0, 0.5, 0.0),
        Quat::IDENTITY,
        sphere(0.5),
    );
    let around = |radius, filter| {
        world.overlap(
            sphere(radius),
            Vec3::new(0.0, 0.5, 0.0),
            Quat::IDENTITY,
            filter,
        )
    };
    assert_eq!(around(1.0, QueryFilter::ALL), vec![floor, a, b]);
    assert_eq!(around(1.0, QueryFilter::ALL.excluding(floor)), vec![a, b]);
    // Radius 0.5 exactly touches the floor and both bodies; touching doesn't count.
    assert!(around(0.5, QueryFilter::ALL).is_empty());
    assert!(
        world
            .overlap(
                Collider::Plane { normal: Vec3::Y },
                Vec3::ZERO,
                Quat::IDENTITY,
                QueryFilter::ALL
            )
            .is_empty()
    );
}

#[test]
fn blink_stops_at_the_wall() {
    // The mage's Blink: cast the character's capsule 15 m forward and stop
    // short of whatever is in the way.
    let mut world = World::default();
    fixed(
        &mut world,
        Vec3::ZERO,
        Quat::IDENTITY,
        Collider::Plane { normal: Vec3::Y },
    );
    let wall = fixed(
        &mut world,
        Vec3::new(0.0, 2.0, -8.0),
        Quat::IDENTITY,
        cuboid(Vec3::new(5.0, 2.0, 0.5)),
    );
    let character = capsule(0.55, 0.35);
    let feet_clear = Vec3::new(0.0, 0.35 + 0.55 + 0.05, 0.0);
    let hit = world
        .shape_cast(
            character,
            feet_clear,
            Quat::IDENTITY,
            Vec3::NEG_Z,
            15.0,
            QueryFilter::ALL,
        )
        .unwrap();
    assert_eq!(hit.body, wall);
    assert!(
        (hit.distance - (7.5 - 0.35)).abs() < EPS,
        "{}",
        hit.distance
    );
}
