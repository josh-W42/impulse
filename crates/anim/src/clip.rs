//! Clips bound to one skeleton, and sampling them at a time.

use glam::Vec3;
use impulse_assets::{Channel, Clip, Interpolation, Keyframes, Skeleton};

use crate::Pose;

/// A clip whose channels point at a skeleton's joints.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundClip {
    pub name: String,
    pub duration: f32,
    /// Channels whose target names no joint of the skeleton (dropped).
    pub unmatched: usize,
    channels: Vec<(usize, Channel)>,
}

impl BoundClip {
    /// Matches each channel to the joint with its target's name.
    pub fn bind(clip: &Clip, skeleton: &Skeleton) -> BoundClip {
        let mut unmatched = 0;
        let channels = clip
            .channels
            .iter()
            .filter_map(|c| match skeleton.find(&c.target) {
                Some(joint) => Some((joint, c.clone())),
                None => {
                    unmatched += 1;
                    None
                }
            })
            .collect();
        BoundClip {
            name: clip.name.clone(),
            duration: clip.duration,
            unmatched,
            channels,
        }
    }

    /// Writes the clip's values at `time` (clamped to the clip) into the
    /// joints it animates; other joints keep what `pose` had.
    pub fn sample(&self, time: f32, pose: &mut Pose) {
        let time = time.clamp(0.0, self.duration);
        for (joint, channel) in &self.channels {
            let Some(local) = pose.local.get_mut(*joint) else {
                continue;
            };
            let Some((i, t)) = keys(&channel.times, time, channel.interpolation) else {
                continue;
            };
            match &channel.keyframes {
                Keyframes::Translation(v) => local.translation = lerp3(v, i, t),
                Keyframes::Scale(v) => local.scale = lerp3(v, i, t),
                Keyframes::Rotation(v) => {
                    let a = v[i];
                    local.rotation = match v.get(i + 1) {
                        Some(&b) if t > 0.0 => a.slerp(b, t),
                        _ => a,
                    }
                    .normalize();
                }
            }
        }
    }
}

/// The key at or before `time` and how far toward the next one (0 for step
/// interpolation or past the last key).
fn keys(times: &[f32], time: f32, interpolation: Interpolation) -> Option<(usize, f32)> {
    if times.is_empty() {
        return None;
    }
    let i = times.partition_point(|&k| k <= time).saturating_sub(1);
    let Some(&next) = times.get(i + 1) else {
        return Some((i, 0.0));
    };
    if interpolation == Interpolation::Step || time <= times[i] {
        return Some((i, 0.0));
    }
    let span = next - times[i];
    let t = if span > 0.0 {
        (time - times[i]) / span
    } else {
        0.0
    };
    Some((i, t.clamp(0.0, 1.0)))
}

fn lerp3(v: &[Vec3], i: usize, t: f32) -> Vec3 {
    match v.get(i + 1) {
        Some(&b) if t > 0.0 => v[i].lerp(b, t),
        _ => v[i],
    }
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};
    use impulse_assets::{
        Channel, Clip, Interpolation, Joint, JointTransform, Keyframes, Skeleton,
    };

    use super::*;
    use crate::Pose;

    fn skeleton(names: &[&str]) -> Skeleton {
        Skeleton {
            joints: names
                .iter()
                .enumerate()
                .map(|(i, name)| Joint {
                    name: name.to_string(),
                    parent: i.checked_sub(1),
                    rest: JointTransform::default(),
                    base: glam::Mat4::IDENTITY,
                    inverse_bind: glam::Mat4::IDENTITY,
                })
                .collect(),
        }
    }

    fn clip(channels: Vec<Channel>) -> Clip {
        let duration = channels
            .iter()
            .map(|c| *c.times.last().unwrap())
            .fold(0.0, f32::max);
        Clip {
            name: "test".into(),
            duration,
            channels,
        }
    }

    fn slide(target: &str, interpolation: Interpolation) -> Channel {
        Channel {
            target: target.into(),
            times: vec![0.0, 1.0],
            keyframes: Keyframes::Translation(vec![Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)]),
            interpolation,
        }
    }

    fn x_at(bound: &BoundClip, skeleton: &Skeleton, time: f32) -> f32 {
        let mut pose = Pose::rest(skeleton);
        bound.sample(time, &mut pose);
        pose.local[0].translation.x
    }

    #[test]
    fn linear_sampling_interpolates_between_keys() {
        let s = skeleton(&["hips"]);
        let bound = BoundClip::bind(&clip(vec![slide("hips", Interpolation::Linear)]), &s);
        assert_eq!(x_at(&bound, &s, 0.25), 0.5);
    }

    #[test]
    fn step_sampling_holds_the_previous_key() {
        let s = skeleton(&["hips"]);
        let bound = BoundClip::bind(&clip(vec![slide("hips", Interpolation::Step)]), &s);
        assert_eq!(x_at(&bound, &s, 0.99), 0.0);
        assert_eq!(x_at(&bound, &s, 1.0), 2.0);
    }

    #[test]
    fn sampling_clamps_outside_the_clip() {
        let s = skeleton(&["hips"]);
        let bound = BoundClip::bind(&clip(vec![slide("hips", Interpolation::Linear)]), &s);
        assert_eq!(x_at(&bound, &s, -1.0), 0.0);
        assert_eq!(x_at(&bound, &s, 5.0), 2.0);
    }

    #[test]
    fn rotation_takes_the_short_way() {
        let s = skeleton(&["hips"]);
        let (a, b) = (Quat::from_rotation_y(0.1), Quat::from_rotation_y(0.3));
        let channel = Channel {
            target: "hips".into(),
            times: vec![0.0, 1.0],
            // −b is the same rotation as b, stored the long way round.
            keyframes: Keyframes::Rotation(vec![a, -b]),
            interpolation: Interpolation::Linear,
        };
        let bound = BoundClip::bind(&clip(vec![channel]), &s);
        let mut pose = Pose::rest(&s);
        bound.sample(0.5, &mut pose);
        let got = pose.local[0].rotation;
        let want = a.slerp(b, 0.5);
        assert!(got.dot(want).abs() > 1.0 - 1e-4, "{got} vs {want}");
    }

    #[test]
    fn unmatched_channels_are_counted() {
        let s = skeleton(&["hips"]);
        let bound = BoundClip::bind(
            &clip(vec![
                slide("hips", Interpolation::Linear),
                slide("tail", Interpolation::Linear),
            ]),
            &s,
        );
        assert_eq!(bound.unmatched, 1);
        assert_eq!(bound.duration, 1.0);
    }

    #[test]
    fn joints_without_channels_keep_their_values() {
        let s = skeleton(&["hips", "spine"]);
        let bound = BoundClip::bind(&clip(vec![slide("hips", Interpolation::Linear)]), &s);
        let mut pose = Pose::rest(&s);
        pose.local[1].translation = Vec3::Y;
        bound.sample(0.5, &mut pose);
        assert_eq!(pose.local[1].translation, Vec3::Y);
    }
}
