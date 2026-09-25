//! The M1 demo scene: bodies dropped from staggered heights, plus how to draw them.

use glam::{Mat4, Quat, Vec3};
use impulse_physics::{BodyDesc, BodyHandle, World};
use impulse_render::{DrawList, MeshKind};

use crate::pose::{Pose, interpolate};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape {
    Box { half_extents: Vec3 },
    Sphere { radius: f32 },
}

impl Shape {
    fn mesh(self) -> MeshKind {
        match self {
            Shape::Box { .. } => MeshKind::Cube,
            Shape::Sphere { .. } => MeshKind::Sphere,
        }
    }

    /// Scale applied to the unit mesh (cube size 1, sphere radius 0.5).
    fn scale(self) -> Vec3 {
        match self {
            Shape::Box { half_extents } => half_extents * 2.0,
            Shape::Sphere { radius } => Vec3::splat(radius * 2.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneObject {
    pub handle: BodyHandle,
    pub shape: Shape,
    pub color: [f32; 4],
    /// Pose before the most recent physics step, for interpolation.
    pub prev: Pose,
}

const PALETTE: [[f32; 4]; 6] = [
    [0.80, 0.25, 0.20, 1.0],
    [0.95, 0.60, 0.15, 1.0],
    [0.20, 0.60, 0.30, 1.0],
    [0.15, 0.45, 0.80, 1.0],
    [0.55, 0.30, 0.75, 1.0],
    [0.85, 0.80, 0.70, 1.0],
];
const FLOOR_SIZE: f32 = 40.0;
const FLOOR_COLOR: [f32; 4] = [0.32, 0.33, 0.35, 1.0];
const GRID_COLOR: [f32; 4] = [0.12, 0.13, 0.15, 1.0];
pub const BODY_COUNT: usize = 30;

/// Replaces everything in `world` with the demo bodies. Deterministic.
pub fn demo_scene(world: &mut World) -> Vec<SceneObject> {
    world.clear();
    (0..BODY_COUNT)
        .map(|i| {
            let (col, row) = ((i % 5) as f32, (i / 5) as f32);
            let position = Vec3::new(col * 2.5 - 5.0, 5.0 + i as f32 * 0.7, row * 2.5 - 6.25);
            let orientation = Quat::from_rotation_y(i as f32 * 0.4);
            let variation = (i % 3) as f32 * 0.1;
            let shape = if i % 2 == 0 {
                Shape::Box {
                    half_extents: Vec3::new(0.5 + variation, 0.5, 0.6 - variation),
                }
            } else {
                Shape::Sphere {
                    radius: 0.4 + variation,
                }
            };
            let desc = BodyDesc {
                orientation,
                ..BodyDesc::dynamic(1.0, position)
            };
            let handle = world.add_body(desc).expect("demo bodies are valid");
            SceneObject {
                handle,
                shape,
                color: PALETTE[i % PALETTE.len()],
                prev: Pose {
                    position,
                    orientation,
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

/// True once every body has fallen below `y` (the demo then re-drops them).
pub fn all_below(world: &World, objects: &[SceneObject], y: f32) -> bool {
    objects
        .iter()
        .filter_map(|o| current_pose(world, o.handle))
        .all(|pose| pose.position.y < y)
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
    list.push_instance(
        MeshKind::Plane,
        Mat4::from_scale(Vec3::new(FLOOR_SIZE, 1.0, FLOOR_SIZE)),
        FLOOR_COLOR,
    );
    for object in objects {
        let Some(current) = current_pose(world, object.handle) else {
            continue;
        };
        let pose = interpolate(object.prev, current, alpha);
        let transform = Mat4::from_scale_rotation_translation(
            object.shape.scale(),
            pose.orientation,
            pose.position,
        );
        list.push_instance(object.shape.mesh(), transform, object.color);
        if show_axes {
            let length = object.shape.scale().max_element();
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

    #[test]
    fn demo_scene_is_deterministic_and_replaces_old_bodies() {
        let mut world = World::default();
        let first = demo_scene(&mut world);
        let second = demo_scene(&mut world);
        assert_eq!(world.len(), BODY_COUNT);
        let poses = |objs: &[SceneObject]| objs.iter().map(|o| o.prev).collect::<Vec<_>>();
        assert_eq!(poses(&first), poses(&second));
        assert!(
            first.iter().all(|o| world.body(o.handle).is_none()),
            "old handles must be stale"
        );
        assert!(
            second.iter().all(|o| o.prev.position.y > 1.0),
            "bodies start above the floor"
        );
    }

    #[test]
    fn draw_list_has_floor_and_every_body_at_the_interpolated_pose() {
        let mut world = World::default();
        let mut objects = demo_scene(&mut world);
        snapshot(&world, &mut objects);
        world.step(0.1);
        let mut list = DrawList::default();
        build_draw_list(&mut list, &world, &objects, 0.0, true, false);
        assert_eq!(list.instances.len(), 1 + BODY_COUNT);
        assert_eq!(list.instances[0].mesh, MeshKind::Plane);
        // alpha 0 draws the pre-step pose, not the current one.
        let drawn_y = list.instances[1].transform.w_axis.y;
        assert_eq!(drawn_y, objects[0].prev.position.y);
        assert_eq!(list.lines.len(), 42);
    }

    #[test]
    fn all_below_detects_a_fallen_scene() {
        let mut world = World::default();
        let objects = demo_scene(&mut world);
        assert!(!all_below(&world, &objects, -30.0));
        for _ in 0..600 {
            world.step(1.0 / 60.0);
        }
        assert!(all_below(&world, &objects, -30.0));
    }
}
