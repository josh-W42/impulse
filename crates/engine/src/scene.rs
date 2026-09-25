//! The M2 demo scene: spheres of varied size and bounciness dropped into a
//! walled pit, plus how to draw it. Render sizes come from the physics colliders.

use glam::{Mat4, Vec3};
use impulse_physics::{BodyDesc, BodyHandle, Collider, Material, World};
use impulse_render::{DrawList, MeshKind};

use crate::pose::{Pose, interpolate};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneObject {
    pub handle: BodyHandle,
    pub color: [f32; 4],
    /// Pose before the most recent physics step, for interpolation.
    pub prev: Pose,
}

/// Alternating spheres and boxes.
pub const BODY_COUNT: usize = 30;
/// Inner half-width of the pit (walls are planes at ±PIT).
const PIT: f32 = 6.0;
const WALL_HEIGHT: f32 = 2.0;
const WALL_THICKNESS: f32 = 0.2;
const FLOOR_SIZE: f32 = 40.0;
const FLOOR_COLOR: [f32; 4] = [0.32, 0.33, 0.35, 1.0];
const WALL_COLOR: [f32; 4] = [0.42, 0.47, 0.55, 1.0];
const GRID_COLOR: [f32; 4] = [0.12, 0.13, 0.15, 1.0];
const LOW_BOUNCE: [f32; 3] = [0.15, 0.40, 0.80];
const HIGH_BOUNCE: [f32; 3] = [0.95, 0.55, 0.15];
const STATIC_MATERIAL: Material = Material {
    restitution: 0.4,
    friction: 0.5,
};

/// Fractional part of `i · k`: a cheap, deterministic spread of values in [0, 1).
fn spread(i: usize, k: f32) -> f32 {
    (i as f32 * k).fract()
}

/// The floor plane and the four inward-facing wall planes: `(point, normal)`.
fn pit_planes() -> [(Vec3, Vec3); 5] {
    [
        (Vec3::ZERO, Vec3::Y),
        (Vec3::new(-PIT, 0.0, 0.0), Vec3::X),
        (Vec3::new(PIT, 0.0, 0.0), Vec3::NEG_X),
        (Vec3::new(0.0, 0.0, -PIT), Vec3::Z),
        (Vec3::new(0.0, 0.0, PIT), Vec3::NEG_Z),
    ]
}

/// Replaces everything in `world` with the pit and its balls. Deterministic.
pub fn demo_scene(world: &mut World) -> Vec<SceneObject> {
    world.clear();
    for (point, normal) in pit_planes() {
        world
            .add_body(
                BodyDesc::fixed(point)
                    .with_collider(Collider::Plane { normal })
                    .with_material(STATIC_MATERIAL),
            )
            .expect("pit planes are valid");
    }
    (0..BODY_COUNT)
        .map(|i| {
            let radius = 0.3 + 0.4 * spread(i, 0.37);
            let restitution = 0.1 + 0.8 * spread(i, 0.61);
            let position = Vec3::new(
                (i % 5) as f32 * 2.0 - 4.0,
                3.0 + i as f32 * 0.4,
                (i / 5) as f32 * 1.6 - 4.0,
            );
            let velocity = Vec3::new(
                spread(i, 0.23) * 4.0 - 2.0,
                0.0,
                spread(i, 0.79) * 4.0 - 2.0,
            );
            // Even: sphere. Odd: box of similar size, spawned tilted and spinning.
            let (collider, mass, orientation, spin) = if i % 2 == 0 {
                (
                    Collider::Sphere { radius },
                    8.0 * radius.powi(3),
                    glam::Quat::IDENTITY,
                    Vec3::ZERO,
                )
            } else {
                let half_extents =
                    Vec3::new(radius, radius * (0.6 + 0.6 * spread(i, 0.43)), radius * 0.8);
                let axis = Vec3::new(spread(i, 0.17) - 0.5, 1.0, spread(i, 0.29) - 0.5).normalize();
                (
                    Collider::Box { half_extents },
                    8.0 * half_extents.x * half_extents.y * half_extents.z,
                    glam::Quat::from_axis_angle(axis, spread(i, 0.53) * std::f32::consts::TAU),
                    Vec3::new(
                        spread(i, 0.31) - 0.5,
                        spread(i, 0.47) - 0.5,
                        spread(i, 0.71) - 0.5,
                    ) * 4.0,
                )
            };
            let desc = BodyDesc {
                orientation,
                ..BodyDesc::dynamic(mass, position)
            }
            .with_collider(collider)
            .with_angular_velocity(spin)
            .with_material(Material {
                restitution,
                friction: 0.4,
            })
            .with_velocity(velocity);
            let handle = world.add_body(desc).expect("demo bodies are valid");
            let t = (restitution - 0.1) / 0.8;
            let c = |k: usize| LOW_BOUNCE[k] + (HIGH_BOUNCE[k] - LOW_BOUNCE[k]) * t;
            SceneObject {
                handle,
                color: [c(0), c(1), c(2), 1.0],
                prev: Pose {
                    position,
                    orientation: desc.orientation,
                },
            }
        })
        .collect()
}

pub fn current_pose(world: &World, handle: BodyHandle) -> Option<Pose> {
    world.body(handle).map(|b| Pose {
        position: b.position(),
        orientation: b.orientation(),
    })
}

/// Records each body's pose; call right before stepping physics.
pub fn snapshot(world: &World, objects: &mut [SceneObject]) {
    for object in objects {
        if let Some(pose) = current_pose(world, object.handle) {
            object.prev = pose;
        }
    }
}

/// True once every body has fallen below `y` (a safety net: the demo re-drops them).
pub fn all_below(world: &World, objects: &[SceneObject], y: f32) -> bool {
    objects
        .iter()
        .filter_map(|o| current_pose(world, o.handle))
        .all(|pose| pose.position.y < y)
}

/// Mesh and scale for drawing a collider; `None` for planes (drawn as scenery).
fn collider_visual(collider: Collider) -> Option<(MeshKind, Vec3)> {
    match collider {
        Collider::Sphere { radius } => Some((MeshKind::Sphere, Vec3::splat(radius * 2.0))),
        Collider::Box { half_extents } => Some((MeshKind::Cube, half_extents * 2.0)),
        Collider::Plane { .. } => None,
    }
}

fn push_scenery(list: &mut DrawList) {
    list.push_instance(
        MeshKind::Plane,
        Mat4::from_scale(Vec3::new(FLOOR_SIZE, 1.0, FLOOR_SIZE)),
        FLOOR_COLOR,
    );
    // Thin boxes just outside each wall plane, so their inner faces sit on it.
    let length = 2.0 * (PIT + WALL_THICKNESS);
    let offset = PIT + WALL_THICKNESS / 2.0;
    let y = WALL_HEIGHT / 2.0;
    for (center, size) in [
        (
            Vec3::new(-offset, y, 0.0),
            Vec3::new(WALL_THICKNESS, WALL_HEIGHT, length),
        ),
        (
            Vec3::new(offset, y, 0.0),
            Vec3::new(WALL_THICKNESS, WALL_HEIGHT, length),
        ),
        (
            Vec3::new(0.0, y, -offset),
            Vec3::new(length, WALL_HEIGHT, WALL_THICKNESS),
        ),
        (
            Vec3::new(0.0, y, offset),
            Vec3::new(length, WALL_HEIGHT, WALL_THICKNESS),
        ),
    ] {
        list.push_instance(
            MeshKind::Cube,
            Mat4::from_scale_rotation_translation(size, glam::Quat::IDENTITY, center),
            WALL_COLOR,
        );
    }
}

pub fn build_draw_list(
    list: &mut DrawList,
    world: &World,
    objects: &[SceneObject],
    alpha: f32,
    show_grid: bool,
    show_axes: bool,
) {
    list.clear();
    push_scenery(list);
    for object in objects {
        let Some(body) = world.body(object.handle) else {
            continue;
        };
        let Some((mesh, scale)) = body.collider().and_then(collider_visual) else {
            continue;
        };
        let current = Pose {
            position: body.position(),
            orientation: body.orientation(),
        };
        let pose = interpolate(object.prev, current, alpha);
        let transform =
            Mat4::from_scale_rotation_translation(scale, pose.orientation, pose.position);
        list.push_instance(mesh, transform, object.color);
        if show_axes {
            let length = scale.max_element();
            for (axis, color) in [
                (Vec3::X, [1.0, 0.2, 0.2, 1.0]),
                (Vec3::Y, [0.2, 1.0, 0.2, 1.0]),
                (Vec3::Z, [0.2, 0.4, 1.0, 1.0]),
            ] {
                let tip = pose.position + pose.orientation * axis * length;
                list.push_line(pose.position, tip, color);
            }
        }
    }
    if show_grid {
        let half = FLOOR_SIZE / 2.0;
        let y = 0.01; // just above the floor to avoid z-fighting
        for i in 0..=20 {
            let t = -half + i as f32 * 2.0;
            list.push_line(Vec3::new(t, y, -half), Vec3::new(t, y, half), GRID_COLOR);
            list.push_line(Vec3::new(-half, y, t), Vec3::new(half, y, t), GRID_COLOR);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCENERY: usize = 5; // floor + 4 walls

    #[test]
    fn pit_has_five_static_planes_and_bodies_start_inside() {
        let mut world = World::default();
        let objects = demo_scene(&mut world);
        assert_eq!(world.len(), 5 + BODY_COUNT);
        let statics = world
            .bodies()
            .filter(|(_, b)| b.inverse_mass() == 0.0)
            .count();
        assert_eq!(statics, 5);
        for o in &objects {
            let body = world.body(o.handle).unwrap();
            // Bounding radius: sphere radius, or the box's half-diagonal.
            let radius = match body.collider() {
                Some(Collider::Sphere { radius }) => radius,
                Some(Collider::Box { half_extents }) => half_extents.length(),
                other => panic!("unexpected collider {other:?}"),
            };
            let p = body.position();
            assert!(
                p.x.abs() + radius < PIT && p.z.abs() + radius < PIT,
                "outside the pit: {p}"
            );
            assert!(p.y - radius > 0.0, "starts in the floor: {p}");
        }
    }

    #[test]
    fn demo_scene_is_deterministic_and_replaces_old_bodies() {
        let mut world = World::default();
        let first = demo_scene(&mut world);
        let second = demo_scene(&mut world);
        assert_eq!(world.len(), 5 + BODY_COUNT);
        let poses = |objs: &[SceneObject]| objs.iter().map(|o| o.prev).collect::<Vec<_>>();
        assert_eq!(poses(&first), poses(&second));
        assert!(
            first.iter().all(|o| world.body(o.handle).is_none()),
            "old handles must be stale"
        );
    }

    #[test]
    fn draw_list_has_scenery_and_every_ball_at_the_interpolated_pose() {
        let mut world = World::default();
        let mut objects = demo_scene(&mut world);
        snapshot(&world, &mut objects);
        world.step(0.1);
        let mut list = DrawList::default();
        build_draw_list(&mut list, &world, &objects, 0.0, true, false);
        assert_eq!(list.instances.len(), SCENERY + BODY_COUNT);
        assert_eq!(list.instances[0].mesh, MeshKind::Plane);
        // alpha 0 draws the pre-step pose, not the current one.
        assert_eq!(
            list.instances[SCENERY].transform.w_axis.y,
            objects[0].prev.position.y
        );
        assert_eq!(list.lines.len(), 42);
    }

    #[test]
    fn bodies_settle_inside_the_pit() {
        let mut world = World::default();
        let objects = demo_scene(&mut world);
        for _ in 0..(8 * 60) {
            world.step(1.0 / 60.0);
        }
        assert!(!all_below(&world, &objects, -30.0));
        for o in &objects {
            let p = world.body(o.handle).unwrap().position();
            assert!(
                p.is_finite() && p.y > -0.1 && p.x.abs() < PIT + 0.1 && p.z.abs() < PIT + 0.1,
                "{p}"
            );
        }
    }
}
