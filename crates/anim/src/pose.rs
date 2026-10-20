//! Poses: one local transform per joint, and what they become in model space.

use glam::Mat4;
use impulse_assets::{JointTransform, Skeleton};

/// Every joint's transform relative to its parent.
#[derive(Debug, Clone, PartialEq)]
pub struct Pose {
    pub local: Vec<JointTransform>,
}

impl Pose {
    /// The skeleton as the file left it.
    pub fn rest(skeleton: &Skeleton) -> Pose {
        Pose {
            local: skeleton.joints.iter().map(|j| j.rest).collect(),
        }
    }
}

/// `out` = `a` at `t` = 0 to `b` at `t` = 1, joint by joint: translation and
/// scale interpolate linearly, rotation along the shorter arc. The three
/// poses must belong to the same skeleton.
pub fn blend(a: &Pose, b: &Pose, t: f32, out: &mut Pose) {
    out.local.clear();
    out.local
        .extend(a.local.iter().zip(&b.local).map(|(a, b)| JointTransform {
            translation: a.translation.lerp(b.translation, t),
            rotation: a.rotation.slerp(b.rotation, t),
            scale: a.scale.lerp(b.scale, t),
        }));
}

/// Each joint's transform in model space: its parent's (or, for a root, its
/// `base`) times its own.
pub fn model_transforms(skeleton: &Skeleton, pose: &Pose, out: &mut Vec<Mat4>) {
    out.clear();
    for (joint, local) in skeleton.joints.iter().zip(&pose.local) {
        let parent = match joint.parent {
            Some(p) => out[p],
            None => joint.base,
        };
        out.push(parent * local.to_mat4());
    }
}

/// What a skinned mesh's vertices are multiplied by: each joint's model
/// transform times its inverse bind matrix.
pub fn skinning_matrices(skeleton: &Skeleton, model: &[Mat4], out: &mut Vec<Mat4>) {
    out.clear();
    out.extend(
        skeleton
            .joints
            .iter()
            .zip(model)
            .map(|(joint, m)| *m * joint.inverse_bind),
    );
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};
    use impulse_assets::{Joint, JointTransform, Skeleton};

    use super::*;

    /// A chain of joints, each 1 m above its parent.
    fn chain(n: usize, base: Mat4) -> Skeleton {
        let rest = JointTransform {
            translation: Vec3::Y,
            ..Default::default()
        };
        let mut joints: Vec<Joint> = (0..n)
            .map(|i| Joint {
                name: format!("j{i}"),
                parent: i.checked_sub(1),
                rest,
                base: if i == 0 { base } else { Mat4::IDENTITY },
                inverse_bind: Mat4::IDENTITY,
            })
            .collect();
        // Bind pose = rest pose.
        let mut model = Mat4::IDENTITY;
        for (i, joint) in joints.iter_mut().enumerate() {
            model = if i == 0 { base } else { model } * rest.to_mat4();
            joint.inverse_bind = model.inverse();
        }
        Skeleton { joints }
    }

    #[test]
    fn rest_pose_with_bind_inverses_gives_identity() {
        let s = chain(3, Mat4::IDENTITY);
        let pose = Pose::rest(&s);
        let (mut model, mut skin) = (Vec::new(), Vec::new());
        model_transforms(&s, &pose, &mut model);
        skinning_matrices(&s, &model, &mut skin);
        assert_eq!(skin.len(), 3);
        for m in skin {
            assert!(m.abs_diff_eq(Mat4::IDENTITY, 1e-5), "{m}");
        }
    }

    #[test]
    fn base_applies_to_root_joints_only() {
        let s = chain(2, Mat4::from_scale(Vec3::splat(100.0)));
        let mut model = Vec::new();
        model_transforms(&s, &Pose::rest(&s), &mut model);
        assert!(
            model[0]
                .w_axis
                .abs_diff_eq(glam::Vec4::new(0.0, 100.0, 0.0, 1.0), 1e-4)
        );
        assert!(
            model[1]
                .w_axis
                .abs_diff_eq(glam::Vec4::new(0.0, 200.0, 0.0, 1.0), 1e-3)
        );
    }

    #[test]
    fn blend_halfway() {
        let s = chain(1, Mat4::IDENTITY);
        let a = Pose::rest(&s);
        let mut b = Pose::rest(&s);
        b.local[0].translation = Vec3::new(2.0, 1.0, 0.0);
        let mut out = Pose::rest(&s);
        blend(&a, &b, 0.5, &mut out);
        assert_eq!(out.local[0].translation, Vec3::new(1.0, 1.0, 0.0));
    }
}
