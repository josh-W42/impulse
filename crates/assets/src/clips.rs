//! Animation clips: keyframes for named nodes, independent of any mesh.
//!
//! Channels name their target node rather than its index, so a clip from one
//! file can drive a skeleton loaded from another (asset packs ship clips and
//! characters separately, with joints in different orders).

use glam::{Quat, Vec3};
use gltf::animation::util::ReadOutputs;
use gltf::animation::{Interpolation as GltfInterpolation, Property};
use gltf::buffer::Source as BufferSource;

use crate::{Limits, ModelError, open};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Linear,
    /// Hold each key until the next.
    Step,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Keyframes {
    Translation(Vec<Vec3>),
    Rotation(Vec<Quat>),
    Scale(Vec<Vec3>),
}

impl Keyframes {
    pub fn len(&self) -> usize {
        match self {
            Self::Translation(v) | Self::Scale(v) => v.len(),
            Self::Rotation(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Channel {
    /// The animated node's name (`#<index>` when it has none).
    pub target: String,
    /// Seconds, increasing; one per keyframe.
    pub times: Vec<f32>,
    pub keyframes: Keyframes,
    pub interpolation: Interpolation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Clip {
    pub name: String,
    /// The last keyframe's time.
    pub duration: f32,
    pub channels: Vec<Channel>,
}

/// Every animation in a `.glb`. The file needs no meshes.
pub fn load_clips(bytes: &[u8]) -> Result<Vec<Clip>, ModelError> {
    let gltf = open(bytes, &Limits::default())?;
    read_clips(&gltf)
}

/// - Morph-target weight channels are skipped.
/// - Cubic-spline channels keep each key's value and play linearly.
pub(crate) fn read_clips(gltf: &gltf::Gltf) -> Result<Vec<Clip>, ModelError> {
    let blob = gltf.blob.as_deref();
    let bin = |buffer: gltf::Buffer| match buffer.source() {
        BufferSource::Bin => blob,
        BufferSource::Uri(_) => None,
    };
    let mut clips = Vec::new();
    for (index, animation) in gltf.animations().enumerate() {
        let name = animation
            .name()
            .map_or_else(|| format!("#{index}"), str::to_owned);
        let mut channels = Vec::new();
        for channel in animation.channels() {
            let node = channel.target().node();
            if channel.target().property() == Property::MorphTargetWeights {
                continue;
            }
            let reader = channel.reader(bin);
            let error = |what: &str| ModelError::Animation(format!("{name}: {what}"));
            let times: Vec<f32> = reader
                .read_inputs()
                .ok_or_else(|| error("a channel has no key times"))?
                .collect();
            if times.iter().any(|t| !t.is_finite()) || times.windows(2).any(|w| w[0] > w[1]) {
                return Err(error("key times must increase"));
            }
            let cubic = channel.sampler().interpolation() == GltfInterpolation::CubicSpline;
            // Cubic splines store (in-tangent, value, out-tangent) per key.
            let pick = |i: usize| !cubic || i % 3 == 1;
            let keyframes = match reader
                .read_outputs()
                .ok_or_else(|| error("a channel has no values"))?
            {
                ReadOutputs::Translations(read) => Keyframes::Translation(
                    read.enumerate()
                        .filter(|(i, _)| pick(*i))
                        .map(|(_, v)| Vec3::from_array(v))
                        .collect(),
                ),
                ReadOutputs::Scales(read) => Keyframes::Scale(
                    read.enumerate()
                        .filter(|(i, _)| pick(*i))
                        .map(|(_, v)| Vec3::from_array(v))
                        .collect(),
                ),
                ReadOutputs::Rotations(read) => Keyframes::Rotation(
                    read.into_f32()
                        .enumerate()
                        .filter(|(i, _)| pick(*i))
                        .map(|(_, q)| Quat::from_array(q).normalize())
                        .collect(),
                ),
                ReadOutputs::MorphTargetWeights(_) => continue,
            };
            if keyframes.len() != times.len() {
                return Err(error("key times and values differ in number"));
            }
            channels.push(Channel {
                target: node
                    .name()
                    .map_or_else(|| format!("#{}", node.index()), str::to_owned),
                times,
                keyframes,
                interpolation: match channel.sampler().interpolation() {
                    GltfInterpolation::Step => Interpolation::Step,
                    GltfInterpolation::Linear | GltfInterpolation::CubicSpline => {
                        Interpolation::Linear
                    }
                },
            });
        }
        let duration = channels
            .iter()
            .filter_map(|c| c.times.last().copied())
            .fold(0.0, f32::max);
        clips.push(Clip {
            name,
            duration,
            channels,
        });
    }
    Ok(clips)
}
