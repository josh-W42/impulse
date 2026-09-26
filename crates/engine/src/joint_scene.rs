//! The M7 "Joints" scene: a swinging capsule chain (ball-sockets), a plank rope
//! bridge (hinges between planks, distance-joint ropes to the posts) with crates
//! dropped on it, and a ragdoll (cone-limited ball-sockets, limited hinges).

use std::f32::consts::FRAC_PI_2;

use glam::{Quat, Vec3};
use impulse_physics::{BodyDesc, BodyHandle, Collider, JointDesc, Material, World};

use crate::pose::Pose;
use crate::scene::SceneObject;

const POST_COLOR: [f32; 4] = [0.45, 0.35, 0.28, 1.0];
const CHAIN_COLOR: [f32; 4] = [0.70, 0.72, 0.78, 1.0];
const PLANK_COLOR: [f32; 4] = [0.78, 0.60, 0.38, 1.0];
const CRATE_COLOR: [f32; 4] = [0.85, 0.35, 0.30, 1.0];
const RAGDOLL_COLOR: [f32; 4] = [0.35, 0.60, 0.85, 1.0];
const HEAD_COLOR: [f32; 4] = [0.95, 0.80, 0.55, 1.0];

pub const CHAIN_LINKS: usize = 8;
pub const PLANKS: usize = 10;
pub const CRATES: usize = 3;
/// Torso, head, 2 × (upper arm, forearm, thigh, shin).
#[cfg(test)]
pub const RAGDOLL_PARTS: usize = 10;

/// Crossbar rope anchors sit this many times wider than the deck.
const ROPE_SPLAY: f32 = 2.0;

const MATERIAL: Material = Material {
    restitution: 0.1,
    friction: 0.6,
};

/// Adds bodies and records them for drawing.
struct Builder<'w> {
    world: &'w mut World,
    objects: Vec<SceneObject>,
}

impl Builder<'_> {
    fn body(&mut self, desc: BodyDesc, color: [f32; 4]) -> BodyHandle {
        let desc = desc.with_material(MATERIAL);
        let handle = self
            .world
            .add_body(desc)
            .expect("joint scene bodies are valid");
        self.objects.push(SceneObject {
            handle,
            color,
            prev: Pose {
                position: desc.position,
                orientation: desc.orientation,
            },
        });
        handle
    }

    fn post(&mut self, at: Vec3, half_extents: Vec3) -> BodyHandle {
        let desc = BodyDesc::fixed(at).with_collider(Collider::Box { half_extents });
        self.body(desc, POST_COLOR)
    }

    fn dynamic(
        &mut self,
        mass: f32,
        at: Vec3,
        orientation: Quat,
        collider: Collider,
        color: [f32; 4],
    ) -> BodyHandle {
        let desc = BodyDesc {
            orientation,
            ..BodyDesc::dynamic(mass, at)
        }
        .with_collider(collider);
        self.body(desc, color)
    }

    fn join(&mut self, desc: JointDesc) {
        self.world
            .add_joint(desc)
            .expect("joint scene joints are valid");
    }
}

pub fn joint_bodies(world: &mut World) -> Vec<SceneObject> {
    let mut b = Builder {
        world,
        objects: Vec::new(),
    };
    chain(&mut b, Vec3::new(-4.0, 5.5, -3.0));
    bridge(&mut b, Vec3::new(0.0, 2.5, 2.5));
    ragdoll(&mut b, Vec3::new(2.5, 2.2, -2.5));
    b.objects
}

/// Capsule links released horizontally, so the chain whips down and swings.
fn chain(b: &mut Builder, top: Vec3) {
    let (half_height, radius) = (0.2, 0.07);
    let pitch = 2.0 * (half_height + radius);
    let mut previous = b.post(top, Vec3::splat(0.15));
    // Capsules are Y-aligned; lay them along +X.
    let along_x = Quat::from_rotation_z(-FRAC_PI_2);
    for i in 0..CHAIN_LINKS {
        let pivot = top + Vec3::X * (0.15 + pitch * i as f32);
        let collider = Collider::Capsule {
            half_height,
            radius,
        };
        let link = b.dynamic(
            0.5,
            pivot + Vec3::X * pitch / 2.0,
            along_x,
            collider,
            CHAIN_COLOR,
        );
        b.join(JointDesc::ball_socket(previous, link, pivot));
        previous = link;
    }
}

/// Hinged planks spanning two posts, hung from them by rope (distance joints)
/// at both corners so the deck can't roll. The ropes splay outward to crossbars
/// wider than the deck, bracing it against swinging sideways, and have enough
/// slack for the deck to sag into a cradle.
fn bridge(b: &mut Builder, center: Vec3) {
    let half = Vec3::new(0.34, 0.05, 0.6);
    let pitch = 2.0 * half.x + 0.04;
    let span = pitch * PLANKS as f32;
    let mut planks = Vec::new();
    for i in 0..PLANKS {
        let x = -span / 2.0 + pitch * (i as f32 + 0.5);
        let plank = b.dynamic(
            1.0,
            center + Vec3::X * x,
            Quat::IDENTITY,
            Collider::Box { half_extents: half },
            PLANK_COLOR,
        );
        if let Some(&previous) = planks.last() {
            let pivot = center + Vec3::X * (x - pitch / 2.0);
            b.join(JointDesc::hinge(previous, plank, pivot, Vec3::Z));
        }
        planks.push(plank);
    }
    for (side, end) in [(-1.0_f32, planks[0]), (1.0, planks[PLANKS - 1])] {
        let post_at = center + Vec3::new(side * (span / 2.0 + 0.6), 0.9, 0.0);
        let post = b.post(post_at, Vec3::new(0.15, 0.15, ROPE_SPLAY * half.z + 0.1));
        for z in [-half.z, half.z] {
            let on_plank = center + Vec3::new(side * span / 2.0, 0.0, z);
            b.join(JointDesc::distance(
                post,
                end,
                post_at + Vec3::Z * (z * ROPE_SPLAY),
                on_plank,
            ));
        }
    }
    for i in 0..CRATES {
        let at = center + Vec3::new(0.8 * (i as f32 - 1.0), 1.5 + i as f32, 0.0);
        b.dynamic(
            1.0,
            at,
            Quat::IDENTITY,
            Collider::Box {
                half_extents: Vec3::splat(0.22),
            },
            CRATE_COLOR,
        );
    }
}

/// A T-posed ragdoll dropped on its feet; front is +Z. Cones bound the neck,
/// shoulders, and hips; elbows and knees are hinges that bend one way only.
fn ragdoll(b: &mut Builder, pelvis: Vec3) {
    let upright = Quat::IDENTITY;
    let torso = b.dynamic(
        3.0,
        pelvis + Vec3::Y * 0.32,
        upright,
        Collider::Box {
            half_extents: Vec3::new(0.22, 0.32, 0.12),
        },
        RAGDOLL_COLOR,
    );
    let head = b.dynamic(
        1.0,
        pelvis + Vec3::Y * 0.83,
        upright,
        Collider::Sphere { radius: 0.14 },
        HEAD_COLOR,
    );
    b.join(JointDesc::ball_socket(torso, head, pelvis + Vec3::Y * 0.66).with_cone(Vec3::Y, 0.6));

    let limb = |half_height: f32, radius: f32| Collider::Capsule {
        half_height,
        radius,
    };
    for side in [-1.0_f32, 1.0] {
        // Arm along ±X; the elbow bends it forward (toward +Z).
        let along_x = Quat::from_rotation_z(-side * FRAC_PI_2);
        let shoulder = pelvis + Vec3::new(side * 0.26, 0.56, 0.0);
        let upper = b.dynamic(
            0.6,
            shoulder + Vec3::X * side * 0.2,
            along_x,
            limb(0.14, 0.06),
            RAGDOLL_COLOR,
        );
        b.join(JointDesc::ball_socket(torso, upper, shoulder).with_cone(Vec3::X * side, 1.3));
        let elbow = shoulder + Vec3::X * side * 0.42;
        let fore = b.dynamic(
            0.5,
            elbow + Vec3::X * side * 0.2,
            along_x,
            limb(0.14, 0.05),
            HEAD_COLOR,
        );
        b.join(JointDesc::hinge(upper, fore, elbow, Vec3::NEG_Y * side).with_limits(0.0, 2.4));

        // Legs hang down; the knee bends the shin backward (toward −Z).
        let hip = pelvis + Vec3::new(side * 0.12, -0.02, 0.0);
        let thigh = b.dynamic(
            1.2,
            hip - Vec3::Y * 0.24,
            upright,
            limb(0.15, 0.08),
            RAGDOLL_COLOR,
        );
        b.join(JointDesc::ball_socket(torso, thigh, hip).with_cone(Vec3::NEG_Y, 1.0));
        let knee = hip - Vec3::Y * 0.48;
        let shin = b.dynamic(
            0.9,
            knee - Vec3::Y * 0.24,
            upright,
            limb(0.15, 0.07),
            RAGDOLL_COLOR,
        );
        b.join(JointDesc::hinge(thigh, shin, knee, Vec3::X).with_limits(0.0, 2.4));
    }
}
