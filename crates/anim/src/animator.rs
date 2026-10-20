//! Playing clips over time: looping or once, at a speed, crossfading from
//! one clip to the next.

use impulse_assets::Skeleton;

use crate::{BoundClip, Pose};

#[derive(Debug, Clone, Copy, PartialEq)]
struct Playback {
    clip: usize,
    time: f32,
    speed: f32,
    looping: bool,
}

impl Playback {
    fn start(clip: usize, looping: bool) -> Self {
        Self {
            clip,
            time: 0.0,
            speed: 1.0,
            looping,
        }
    }

    fn advance(&mut self, dt: f32, clips: &[BoundClip]) {
        let duration = clips.get(self.clip).map_or(0.0, |c| c.duration);
        self.time += dt * self.speed;
        self.time = if self.looping && duration > 0.0 {
            self.time.rem_euclid(duration)
        } else {
            self.time.clamp(0.0, duration)
        };
    }

    fn sample(&self, clips: &[BoundClip], pose: &mut Pose) {
        if let Some(clip) = clips.get(self.clip) {
            clip.sample(self.time, pose);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Fade {
    from: Playback,
    elapsed: f32,
    duration: f32,
}

/// One unit's animation state: the clip playing (an index into the clips
/// it is given) and, during a crossfade, the one it is leaving.
#[derive(Debug, Clone, PartialEq)]
pub struct Animator {
    current: Playback,
    fade: Option<Fade>,
}

impl Animator {
    /// Looping `clip` at normal speed from its start.
    pub fn new(clip: usize) -> Self {
        Self {
            current: Playback::start(clip, true),
            fade: None,
        }
    }

    /// Switches to `clip`, crossfading over `fade` seconds. If `clip` is
    /// already playing it carries on (only `looping` changes).
    pub fn play(&mut self, clip: usize, looping: bool, fade: f32) {
        if clip == self.current.clip {
            self.current.looping = looping;
            return;
        }
        self.restart(clip, looping, fade);
    }

    /// Starts `clip` from the beginning even if it is playing (a second
    /// swing of the same attack), crossfading over `fade` seconds.
    pub fn restart(&mut self, clip: usize, looping: bool, fade: f32) {
        self.fade = (fade > 0.0).then_some(Fade {
            from: self.current,
            elapsed: 0.0,
            duration: fade,
        });
        self.current = Playback::start(clip, looping);
    }

    /// The playing clip's rate (1 is as authored).
    pub fn set_speed(&mut self, speed: f32) {
        self.current.speed = speed;
    }

    pub fn advance(&mut self, dt: f32, clips: &[BoundClip]) {
        self.current.advance(dt, clips);
        if let Some(fade) = &mut self.fade {
            fade.from.advance(dt, clips);
            fade.elapsed += dt;
            if fade.elapsed >= fade.duration {
                self.fade = None;
            }
        }
    }

    pub fn current(&self) -> usize {
        self.current.clip
    }

    /// Seconds into the playing clip.
    pub fn time(&self) -> f32 {
        self.current.time
    }

    /// Sets the time into the playing clip (clamped to it).
    pub fn seek(&mut self, time: f32, clips: &[BoundClip]) {
        self.current.time = time;
        self.current.advance(0.0, clips);
    }

    /// A clip played once has reached its end.
    pub fn finished(&self, clips: &[BoundClip]) -> bool {
        !self.current.looping
            && clips
                .get(self.current.clip)
                .is_none_or(|c| self.current.time >= c.duration)
    }

    /// The pose now, into `out`. `scratch` is working space; both must be
    /// for `skeleton`.
    pub fn pose(
        &self,
        clips: &[BoundClip],
        skeleton: &Skeleton,
        scratch: &mut Pose,
        out: &mut Pose,
    ) {
        *out = Pose::rest(skeleton);
        self.current.sample(clips, out);
        let Some(fade) = &self.fade else { return };
        *scratch = Pose::rest(skeleton);
        fade.from.sample(clips, scratch);
        let w = (fade.elapsed / fade.duration).clamp(0.0, 1.0);
        for (to, from) in out.local.iter_mut().zip(&scratch.local) {
            to.translation = from.translation.lerp(to.translation, w);
            to.rotation = from.rotation.slerp(to.rotation, w);
            to.scale = from.scale.lerp(to.scale, w);
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};
    use impulse_assets::{
        Channel, Clip, Interpolation, Joint, JointTransform, Keyframes, Skeleton,
    };

    use super::*;
    use crate::{BoundClip, Pose};

    fn skeleton() -> Skeleton {
        Skeleton {
            joints: vec![Joint {
                name: "hips".into(),
                parent: None,
                rest: JointTransform::default(),
                base: Mat4::IDENTITY,
                inverse_bind: Mat4::IDENTITY,
            }],
        }
    }

    /// A 1 s clip moving the hips' x from `from` to `to`.
    fn slide(from: f32, to: f32, s: &Skeleton) -> BoundClip {
        let clip = Clip {
            name: format!("{from}->{to}"),
            duration: 1.0,
            channels: vec![Channel {
                target: "hips".into(),
                times: vec![0.0, 1.0],
                keyframes: Keyframes::Translation(vec![Vec3::X * from, Vec3::X * to]),
                interpolation: Interpolation::Linear,
            }],
        };
        BoundClip::bind(&clip, s)
    }

    fn x(animator: &Animator, clips: &[BoundClip], s: &Skeleton) -> f32 {
        let (mut scratch, mut out) = (Pose::rest(s), Pose::rest(s));
        animator.pose(clips, s, &mut scratch, &mut out);
        out.local[0].translation.x
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn looping_wraps() {
        let s = skeleton();
        let clips = [slide(0.0, 2.0, &s)];
        let mut a = Animator::new(0);
        a.advance(2.25, &clips);
        assert!(close(x(&a, &clips, &s), 0.5));
        assert!(!a.finished(&clips));
    }

    #[test]
    fn one_shot_stops_and_reports_finished() {
        let s = skeleton();
        let clips = [slide(0.0, 2.0, &s)];
        let mut a = Animator::new(0);
        a.play(0, false, 0.0);
        a.advance(3.0, &clips);
        assert!(close(x(&a, &clips, &s), 2.0));
        assert!(a.finished(&clips));
    }

    #[test]
    fn crossfade_weights_over_time() {
        let s = skeleton();
        let clips = [slide(0.0, 0.0, &s), slide(2.0, 2.0, &s)];
        let mut a = Animator::new(0);
        a.play(1, true, 0.2);
        assert_eq!(a.current(), 1);
        a.advance(0.1, &clips);
        assert!(close(x(&a, &clips, &s), 1.0), "{}", x(&a, &clips, &s));
        a.advance(0.15, &clips);
        assert_eq!(x(&a, &clips, &s), 2.0, "the fade is over");
    }

    #[test]
    fn play_same_clip_does_not_restart() {
        let s = skeleton();
        let clips = [slide(0.0, 2.0, &s)];
        let mut a = Animator::new(0);
        a.advance(0.5, &clips);
        a.play(0, true, 0.2);
        assert!(close(x(&a, &clips, &s), 1.0));
    }

    #[test]
    fn restart_starts_over() {
        let s = skeleton();
        let clips = [slide(0.0, 2.0, &s)];
        let mut a = Animator::new(0);
        a.advance(0.5, &clips);
        a.restart(0, true, 0.2);
        // Still all the old playback (x = 1) at the start of the fade...
        assert!(close(x(&a, &clips, &s), 1.0));
        // ...and all the new one, 0.2 s in, once it's over.
        a.advance(0.2, &clips);
        assert!(close(x(&a, &clips, &s), 0.4), "{}", x(&a, &clips, &s));
    }

    #[test]
    fn speed_scales_time() {
        let s = skeleton();
        let clips = [slide(0.0, 2.0, &s)];
        let mut a = Animator::new(0);
        a.set_speed(2.0);
        a.advance(0.25, &clips);
        assert!(close(x(&a, &clips, &s), 1.0));
    }
    #[test]
    fn seek_jumps_within_the_clip() {
        let s = skeleton();
        let clips = [slide(0.0, 2.0, &s)];
        let mut a = Animator::new(0);
        a.play(0, false, 0.0);
        a.seek(5.0, &clips);
        assert_eq!(a.time(), 1.0, "clamped to the clip's end");
        assert!(a.finished(&clips));
        a.seek(0.25, &clips);
        assert!(close(x(&a, &clips, &s), 0.5));
    }
}
