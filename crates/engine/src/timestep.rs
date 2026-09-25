//! Fixed-timestep accumulator ("Fix Your Timestep"): physics always advances in
//! equal steps, independent of frame rate; rendering interpolates the remainder.

/// What to do this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepPlan {
    /// Fixed steps to run now.
    pub steps: u32,
    /// Leftover fraction of a step, in `[0, 1)`, for interpolating rendered poses.
    pub alpha: f32,
}

#[derive(Debug, Clone)]
pub struct FixedTimestep {
    step: f32,
    max_steps: u32,
    accumulator: f32,
}

impl FixedTimestep {
    /// `max_steps` caps work per frame so a slow frame can't snowball ("spiral of death").
    pub fn new(step: f32, max_steps: u32) -> Self {
        assert!(step.is_finite() && step > 0.0, "step must be positive");
        assert!(max_steps > 0, "max_steps must be positive");
        Self {
            step,
            max_steps,
            accumulator: 0.0,
        }
    }

    pub fn step(&self) -> f32 {
        self.step
    }

    /// Adds `frame_time` seconds; invalid (negative or non-finite) times count as zero.
    pub fn advance(&mut self, frame_time: f32) -> StepPlan {
        if frame_time.is_finite() && frame_time > 0.0 {
            self.accumulator += frame_time;
        }
        let mut steps = 0;
        while self.accumulator >= self.step && steps < self.max_steps {
            self.accumulator -= self.step;
            steps += 1;
        }
        // Capped: drop the backlog but keep the sub-step remainder.
        if self.accumulator >= self.step {
            self.accumulator %= self.step;
        }
        StepPlan {
            steps,
            alpha: self.accumulator / self.step,
        }
    }

    pub fn reset(&mut self) {
        self.accumulator = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1/64 is exact in binary, so these tests don't depend on float rounding at step boundaries.
    const STEP: f32 = 1.0 / 64.0;

    #[test]
    fn whole_steps_leave_no_remainder() {
        let mut t = FixedTimestep::new(STEP, 5);
        assert_eq!(
            t.advance(3.0 * STEP),
            StepPlan {
                steps: 3,
                alpha: 0.0
            }
        );
    }

    #[test]
    fn partial_steps_accumulate() {
        let mut t = FixedTimestep::new(STEP, 5);
        assert_eq!(
            t.advance(STEP / 2.0),
            StepPlan {
                steps: 0,
                alpha: 0.5
            }
        );
        assert_eq!(
            t.advance(STEP / 2.0),
            StepPlan {
                steps: 1,
                alpha: 0.0
            }
        );
    }

    #[test]
    fn long_frames_are_capped() {
        let mut t = FixedTimestep::new(STEP, 5);
        let plan = t.advance(1.0);
        assert_eq!(plan.steps, 5);
        assert!((0.0..1.0).contains(&plan.alpha));
        // The dropped backlog must not resurface next frame.
        assert_eq!(t.advance(0.0).steps, 0);
    }

    #[test]
    fn invalid_frame_times_are_ignored() {
        let mut t = FixedTimestep::new(STEP, 5);
        for bad in [f32::NAN, f32::INFINITY, -1.0] {
            assert_eq!(
                t.advance(bad),
                StepPlan {
                    steps: 0,
                    alpha: 0.0
                }
            );
        }
    }

    #[test]
    fn alpha_stays_in_unit_range() {
        let mut t = FixedTimestep::new(1.0 / 60.0, 5);
        let mut seed: u32 = 12345;
        for _ in 0..1000 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let frame = (seed >> 8) as f32 / (1u32 << 24) as f32 * 0.1; // 0..100 ms
            let alpha = t.advance(frame).alpha;
            assert!((0.0..1.0).contains(&alpha), "alpha {alpha}");
        }
    }
}
