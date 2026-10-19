//! Kinematic character controller: a capsule moved by code that walks on
//! ground, slides along walls, climbs slopes and steps up to a limit, falls,
//! and jumps. Built on shape casts; the capsule itself is a kinematic body, so
//! it pushes dynamic bodies it isn't set to collide with in queries.
//!
//! Each update plans the move against the world as it is (collide and slide:
//! sweep, stop at the first hit, slide the rest along the surface, repeat),
//! then hands the result to [`World::set_kinematic_target`]; the next
//! [`World::step`] carries the body there. Y is up.

use glam::{Quat, Vec3};

use crate::body::{BodyDesc, BodyError};
use crate::collider::Collider;
use crate::filter::{CollisionFilter, QueryFilter};
use crate::query::ShapeHit;
use crate::world::{BodyHandle, World};

/// Sweeps per move; each slides the remainder along one more surface.
const MAX_SLIDES: usize = 4;
/// Moves shorter than this (metres) are skipped.
const MIN_MOVE: f32 = 1e-5;

/// Shape and movement limits of a character.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacterDesc {
    /// Capsule radius.
    pub radius: f32,
    /// Half the length of the capsule's straight part; standing height is
    /// `2 · (radius + half_height)`.
    pub half_height: f32,
    /// Steepest walkable slope, radians from horizontal. Steeper surfaces are
    /// walls.
    pub max_slope: f32,
    /// Tallest ledge walked onto without jumping (metres).
    pub step_height: f32,
    /// How far below the feet ground is still followed when walking down slopes
    /// and steps, instead of starting to fall.
    pub snap_distance: f32,
    /// Gap kept between the capsule and everything it touches, so sweeps never
    /// start in contact.
    pub skin: f32,
    /// Upward speed (m/s) at the start of a jump.
    pub jump_speed: f32,
    /// The capsule body's collision filter (what dynamic bodies it pushes).
    pub filter: CollisionFilter,
    /// Groups the character walks on and is blocked by.
    pub query_mask: u32,
}

impl Default for CharacterDesc {
    /// Human-sized: 1.8 m tall, 45° slopes, 35 cm steps, about 1.3 m jumps.
    fn default() -> Self {
        Self {
            radius: 0.35,
            half_height: 0.55,
            max_slope: 45f32.to_radians(),
            step_height: 0.35,
            snap_distance: 0.3,
            skin: 0.02,
            jump_speed: 5.0,
            filter: CollisionFilter::DEFAULT,
            query_mask: u32::MAX,
        }
    }
}

impl CharacterDesc {
    fn collider(&self) -> Collider {
        Collider::Capsule {
            half_height: self.half_height,
            radius: self.radius,
        }
    }

    /// From the feet to the capsule's center.
    fn center_offset(&self) -> Vec3 {
        Vec3::Y * (self.radius + self.half_height)
    }

    fn is_valid(&self) -> bool {
        let finite_non_negative = |x: f32| x.is_finite() && x >= 0.0;
        self.radius.is_finite()
            && self.radius > 0.0
            && finite_non_negative(self.half_height)
            && self.max_slope.is_finite()
            && (0.0..std::f32::consts::FRAC_PI_2).contains(&self.max_slope)
            && finite_non_negative(self.step_height)
            && finite_non_negative(self.snap_distance)
            && self.skin.is_finite()
            && self.skin > 0.0
            && finite_non_negative(self.jump_speed)
    }
}

/// A character in a [`World`]: a kinematic capsule plus the state that moving
/// it needs (vertical speed, ground contact).
#[derive(Debug, Clone, PartialEq)]
pub struct CharacterController {
    body: BodyHandle,
    desc: CharacterDesc,
    /// Planned feet position: where the body will be after the next step.
    feet: Vec3,
    vertical_speed: f32,
    grounded: bool,
    ground_normal: Vec3,
}

impl CharacterController {
    /// Adds the character's capsule standing with its feet at `feet`.
    pub fn new(world: &mut World, feet: Vec3, desc: CharacterDesc) -> Result<Self, BodyError> {
        if !desc.is_valid() {
            return Err(BodyError::InvalidCollider);
        }
        let body = world.add_body(
            BodyDesc::kinematic(feet + desc.center_offset())
                .with_collider(desc.collider())
                .with_filter(desc.filter),
        )?;
        Ok(Self {
            body,
            desc,
            feet,
            vertical_speed: 0.0,
            grounded: false,
            ground_normal: Vec3::Y,
        })
    }

    /// The capsule body (for drawing, or for excluding it from other queries).
    pub fn body(&self) -> BodyHandle {
        self.body
    }

    pub fn desc(&self) -> &CharacterDesc {
        &self.desc
    }

    /// Bottom of the capsule, as of the last update (where the body is once the
    /// world has stepped).
    pub fn feet(&self) -> Vec3 {
        self.feet
    }

    /// Standing on walkable ground.
    pub fn is_grounded(&self) -> bool {
        self.grounded
    }

    /// Normal of the ground stood on (`Y` in the air).
    pub fn ground_normal(&self) -> Vec3 {
        self.ground_normal
    }

    /// Vertical speed (m/s, up positive): zero on the ground.
    pub fn vertical_speed(&self) -> f32 {
        self.vertical_speed
    }

    /// Moves the character to `feet` immediately (spawning, respawning,
    /// teleports) and stops its fall.
    pub fn teleport(&mut self, world: &mut World, feet: Vec3) -> bool {
        if !world.set_pose(self.body, feet + self.desc.center_offset(), Quat::IDENTITY) {
            return false;
        }
        world.set_velocity(self.body, Vec3::ZERO);
        self.feet = feet;
        self.vertical_speed = 0.0;
        self.grounded = false;
        true
    }

    /// Plans this step's movement: walk at `velocity` (horizontal m/s; its Y is
    /// ignored), jump if `jump` and on the ground, fall under the world's
    /// gravity otherwise. Call once per step, before [`World::step`] with the
    /// same `dt`.
    pub fn update(&mut self, world: &mut World, velocity: Vec3, jump: bool, dt: f32) {
        if !(dt.is_finite() && dt > 0.0) || !velocity.is_finite() {
            return;
        }
        let mut feet = self.feet;
        let gravity = world.gravity().y;

        let jumped = jump && self.grounded;
        if jumped {
            self.vertical_speed = self.desc.jump_speed;
        } else if self.grounded {
            self.vertical_speed = 0.0;
        } else {
            self.vertical_speed += gravity * dt;
        }

        // Walk, stepping up ledges that block it.
        let walk = Vec3::new(velocity.x, 0.0, velocity.z) * dt;
        if walk.length() > MIN_MOVE {
            feet = self.walk(world, feet, walk);
        }

        // Fall or rise.
        let was_grounded = self.grounded;
        self.grounded = false;
        self.ground_normal = Vec3::Y;
        let rise = self.vertical_speed * dt;
        if rise < -MIN_MOVE {
            // Falling: land on ground that holds the character, slide down
            // anything steeper.
            match self.cast(world, feet, Vec3::NEG_Y, -rise + self.desc.skin) {
                Some(hit) => match self.support(world, feet, &hit) {
                    Some((normal, _)) => {
                        feet.y -= (hit.distance - self.desc.skin).max(0.0);
                        self.land(normal);
                    }
                    None => feet = self.slide(world, feet, Vec3::Y * rise, false).0,
                },
                None => feet.y += rise,
            }
        } else if rise > MIN_MOVE {
            let (moved, hits) = self.slide(world, feet, Vec3::Y * rise, false);
            feet = moved;
            if hits.iter().any(|h| h.normal.y < -0.5) {
                self.vertical_speed = 0.0; // head against a ceiling
            }
        }

        // Follow the ground down slopes and steps instead of launching off.
        if was_grounded
            && !jumped
            && !self.grounded
            && let Some(hit) = self.cast(world, feet, Vec3::NEG_Y, self.desc.snap_distance)
            && let Some((normal, _)) = self.support(world, feet, &hit)
        {
            feet.y -= (hit.distance - self.desc.skin).max(0.0);
            self.land(normal);
        }

        self.feet = feet;
        world.set_kinematic_target(
            self.body,
            feet + self.desc.center_offset(),
            Quat::IDENTITY,
            dt,
        );
    }

    fn land(&mut self, normal: Vec3) {
        self.grounded = true;
        self.ground_normal = normal;
        self.vertical_speed = 0.0;
    }

    /// Horizontal move with step-up: if a wall stops it short, try going up by
    /// the step height, across, and back down onto walkable ground, and keep
    /// whichever got further.
    fn walk(&self, world: &World, feet: Vec3, walk: Vec3) -> Vec3 {
        let (flat, hits) = self.slide(world, feet, walk, true);
        let blocked = hits.iter().any(|h| !self.walkable(h.normal));
        if !blocked || self.desc.step_height <= 0.0 || !self.grounded {
            return flat;
        }
        // Up (as far as the ceiling allows), across, then down.
        let up = self
            .cast(world, feet, Vec3::Y, self.desc.step_height)
            .map_or(self.desc.step_height, |h| {
                (h.distance - self.desc.skin).max(0.0)
            });
        let raised = feet + Vec3::Y * up;
        let (across, _) = self.slide(world, raised, walk, true);
        let Some(down) = self.cast(world, across, Vec3::NEG_Y, up + self.desc.skin) else {
            return flat; // nothing to stand on over there
        };
        // Measured at the surface found: the capsule's rounded bottom can settle
        // on the corner of a ledge taller than the step, with the feet lower.
        match self.support(world, across, &down) {
            Some((_, height)) if height - feet.y <= self.desc.step_height + 1e-3 => {}
            _ => return flat,
        }
        let stepped = across - Vec3::Y * (down.distance - self.desc.skin).max(0.0);
        let progress = |p: Vec3| (p - feet).with_y(0.0).dot(walk);
        if progress(stepped) > progress(flat) + MIN_MOVE {
            stepped
        } else {
            flat
        }
    }

    /// Collide and slide: sweep, stop `skin` short of the first hit, and slide
    /// what's left along the surfaces hit so far. `walking` treats steep
    /// surfaces as vertical walls, so sliding along them can't climb them.
    fn slide(
        &self,
        world: &World,
        mut feet: Vec3,
        motion: Vec3,
        walking: bool,
    ) -> (Vec3, Vec<ShapeHit>) {
        let mut remaining = motion;
        let mut planes: Vec<Vec3> = Vec::new();
        let mut hits = Vec::new();
        for _ in 0..MAX_SLIDES {
            let length = remaining.length();
            if length < MIN_MOVE {
                break;
            }
            let dir = remaining / length;
            let Some(hit) = self.cast(world, feet, dir, length + self.desc.skin) else {
                feet += remaining;
                break;
            };
            // Stop `skin` short measured along the normal, not along the path:
            // backing off `skin` along a diagonal path would creep closer.
            let approach = (-dir.dot(hit.normal)).max(0.1);
            let travel = (hit.distance - self.desc.skin / approach).clamp(0.0, length);
            feet += dir * travel;
            let gap = (hit.distance - travel) * approach;
            if gap < self.desc.skin - 1e-4 {
                // Closer than the skin (pushed in by something): ease back out.
                feet += hit.normal * (self.desc.skin - gap);
            }
            remaining = dir * (length - travel);
            let mut normal = hit.normal;
            if walking && !self.walkable(normal) {
                normal = normal
                    .with_y(0.0)
                    .normalize_or(-dir.with_y(0.0).normalize_or(Vec3::X));
            }
            hits.push(hit);
            planes.push(normal);
            remaining = clip(remaining, &planes);
        }
        (feet, hits)
    }

    fn cast(&self, world: &World, feet: Vec3, dir: Vec3, distance: f32) -> Option<ShapeHit> {
        world.shape_cast(
            self.desc.collider(),
            feet + self.desc.center_offset(),
            Quat::IDENTITY,
            dir,
            distance,
            QueryFilter::mask(self.desc.query_mask).excluding(self.body),
        )
    }

    /// The walkable surface under a downward hit, if any: its normal and
    /// height. A capsule meets a ledge first at its edge, where the contact
    /// normal points mostly sideways; what matters is the surface beyond the
    /// edge, so a short ray just past the contact reads it.
    fn support(&self, world: &World, feet: Vec3, hit: &ShapeHit) -> Option<(Vec3, f32)> {
        if self.walkable(hit.normal) {
            return Some((hit.normal, hit.point.y));
        }
        let center = feet + self.desc.center_offset();
        let outward = (hit.point - center).with_y(0.0).normalize_or_zero();
        if outward == Vec3::ZERO {
            return None;
        }
        const PROBE: f32 = 0.02;
        let ray = world.raycast_filtered(
            hit.point + outward * (PROBE / 2.0) + Vec3::Y * PROBE,
            Vec3::NEG_Y,
            2.0 * PROBE,
            QueryFilter::mask(self.desc.query_mask).excluding(self.body),
        )?;
        self.walkable(ray.normal)
            .then_some((ray.normal, ray.point.y))
    }

    fn walkable(&self, normal: Vec3) -> bool {
        normal.y >= self.desc.max_slope.cos() - 1e-4
    }
}

/// Removes the parts of `motion` that push into any of `planes` (unit
/// normals): slides along one plane, along the crease of two, stops at three.
fn clip(motion: Vec3, planes: &[Vec3]) -> Vec3 {
    let Some(&last) = planes.last() else {
        return motion;
    };
    let along = motion - last * motion.dot(last).min(0.0);
    let into = |m: Vec3| planes.iter().any(|&p| m.dot(p) < -1e-6);
    if !into(along) {
        return along;
    }
    // Sliding along the newest plane runs into an earlier one: follow their crease.
    for &earlier in &planes[..planes.len() - 1] {
        let crease = last.cross(earlier);
        if crease.length_squared() < 1e-10 {
            continue;
        }
        let crease = crease.normalize();
        let along_crease = crease * motion.dot(crease);
        if !into(along_crease) {
            return along_crease;
        }
    }
    Vec3::ZERO
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_slides_along_one_plane() {
        let wall = Vec3::NEG_X;
        let slid = clip(Vec3::new(1.0, 0.0, 1.0), &[wall]);
        assert!(slid.abs_diff_eq(Vec3::new(0.0, 0.0, 1.0), 1e-6));
        // Moving away from a plane isn't changed.
        assert_eq!(
            clip(Vec3::new(-1.0, 0.0, 1.0), &[wall]),
            Vec3::new(-1.0, 0.0, 1.0)
        );
    }

    #[test]
    fn clip_follows_creases_and_stops_in_corners() {
        // Two walls meeting at a right angle (an inside corner), floor below.
        let (a, b, floor) = (Vec3::NEG_X, Vec3::NEG_Z, Vec3::Y);
        assert_eq!(clip(Vec3::new(1.0, 0.0, 1.0), &[a, b]), Vec3::ZERO);
        let crease = clip(Vec3::new(1.0, -1.0, 0.3), &[a, floor]);
        assert!(
            crease.abs_diff_eq(Vec3::new(0.0, 0.0, 0.3), 1e-6),
            "{crease}"
        );
    }

    #[test]
    fn rejects_bad_descriptions() {
        let mut world = World::default();
        for desc in [
            CharacterDesc {
                radius: 0.0,
                ..CharacterDesc::default()
            },
            CharacterDesc {
                max_slope: 2.0,
                ..CharacterDesc::default()
            },
            CharacterDesc {
                skin: 0.0,
                ..CharacterDesc::default()
            },
        ] {
            assert!(CharacterController::new(&mut world, Vec3::ZERO, desc).is_err());
        }
        assert!(world.is_empty());
    }
}
