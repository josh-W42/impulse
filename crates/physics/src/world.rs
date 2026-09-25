use glam::Vec3;

use crate::body::{BodyDesc, BodyError, RigidBody};
use crate::contact::find_contacts;
use crate::solver;

pub const DEFAULT_GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

/// Refers to a body in a [`World`]. Stays safe after removal: a stale handle
/// never reaches a different body that later reuses the same slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BodyHandle {
    index: u32,
    generation: u32,
}

#[derive(Debug, Clone)]
struct Slot {
    generation: u32,
    body: Option<RigidBody>,
}

/// All bodies plus global settings. Bodies live in a generational arena.
#[derive(Debug, Clone)]
pub struct World {
    gravity: Vec3,
    slots: Vec<Slot>,
    free: Vec<u32>,
}

impl Default for World {
    fn default() -> Self {
        Self::new(DEFAULT_GRAVITY)
    }
}

impl World {
    pub fn new(gravity: Vec3) -> Self {
        Self {
            gravity,
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    pub fn gravity(&self) -> Vec3 {
        self.gravity
    }

    pub fn add_body(&mut self, desc: BodyDesc) -> Result<BodyHandle, BodyError> {
        let body = RigidBody::new(desc)?;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.body = Some(body);
            return Ok(BodyHandle {
                index,
                generation: slot.generation,
            });
        }
        let index = u32::try_from(self.slots.len()).expect("more than u32::MAX bodies");
        self.slots.push(Slot {
            generation: 0,
            body: Some(body),
        });
        Ok(BodyHandle {
            index,
            generation: 0,
        })
    }

    pub fn remove_body(&mut self, handle: BodyHandle) -> Option<RigidBody> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        let body = slot.body.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(handle.index);
        Some(body)
    }

    pub fn body(&self, handle: BodyHandle) -> Option<&RigidBody> {
        let slot = self.slots.get(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.body.as_ref()
    }

    pub fn bodies(&self) -> impl Iterator<Item = (BodyHandle, &RigidBody)> {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            let body = slot.body.as_ref()?;
            let handle = BodyHandle {
                index: index as u32,
                generation: slot.generation,
            };
            Some((handle, body))
        })
    }

    pub fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Removes every body; all existing handles become stale.
    pub fn clear(&mut self) {
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if slot.body.take().is_some() {
                slot.generation = slot.generation.wrapping_add(1);
                self.free.push(index as u32);
            }
        }
    }

    /// Advances the simulation by `dt` seconds.
    ///
    /// Order: gravity → find contacts → solve contact velocities → move →
    /// push remaining overlap apart. (Semi-implicit Euler with the contact solve
    /// between the velocity and position updates.)
    pub fn step(&mut self, dt: f32) {
        debug_assert!(dt.is_finite() && dt >= 0.0, "invalid timestep {dt}");
        let gravity = self.gravity;
        for body in self.bodies_mut() {
            body.apply_gravity(gravity, dt);
        }
        let contacts = find_contacts(self);
        solver::solve_velocities(self, &contacts);
        for body in self.bodies_mut() {
            body.advance(dt);
        }
        solver::correct_positions(self);
    }

    fn bodies_mut(&mut self) -> impl Iterator<Item = &mut RigidBody> {
        self.slots.iter_mut().filter_map(|slot| slot.body.as_mut())
    }

    pub(crate) fn body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.body.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use glam::Quat;

    use super::*;
    use crate::body::BodyKind;

    const G: f32 = 9.81;

    fn fall(dt: f32, seconds: f32) -> RigidBody {
        let mut world = World::default();
        let h = world
            .add_body(BodyDesc::dynamic(1.0, Vec3::new(0.0, 100.0, 0.0)))
            .unwrap();
        for _ in 0..(seconds / dt).round() as u32 {
            world.step(dt);
        }
        *world.body(h).unwrap()
    }

    #[test]
    fn free_fall_velocity_is_exact() {
        let body = fall(1.0 / 60.0, 2.0);
        assert!((body.velocity().y - (-G * 2.0)).abs() < 1e-3);
    }

    #[test]
    fn free_fall_position_error_is_first_order() {
        let t = 2.0;
        let exact = 100.0 - 0.5 * G * t * t;
        // Semi-implicit Euler moves with the *updated* velocity, so it falls slightly
        // further than the exact parabola; explicit Euler would lag above it.
        assert!(
            fall(1.0 / 60.0, t).position().y < exact,
            "not semi-implicit Euler"
        );
        let err = |dt: f32| (fall(dt, t).position().y - exact).abs();
        let (coarse, fine) = (err(1.0 / 60.0), err(1.0 / 120.0));
        assert!(coarse <= G * t / 60.0, "error {coarse} too large");
        let ratio = fine / coarse;
        assert!(
            (0.45..0.55).contains(&ratio),
            "halving dt gave ratio {ratio}"
        );
    }

    #[test]
    fn static_bodies_never_move() {
        let mut world = World::default();
        let desc = BodyDesc {
            velocity: Vec3::new(5.0, 5.0, 5.0),
            ..BodyDesc::fixed(Vec3::ONE)
        };
        let h = world.add_body(desc).unwrap();
        for _ in 0..100 {
            world.step(1.0 / 60.0);
        }
        assert_eq!(world.body(h).unwrap().position(), Vec3::ONE);
        assert_eq!(world.body(h).unwrap().inverse_mass(), 0.0);
    }

    #[test]
    fn rejects_invalid_bodies() {
        let mut world = World::default();
        let ok = BodyDesc::dynamic(1.0, Vec3::ZERO);
        let cases = [
            (
                BodyDesc {
                    position: Vec3::NAN,
                    ..ok
                },
                BodyError::NonFinite,
            ),
            (
                BodyDesc {
                    velocity: Vec3::INFINITY,
                    ..ok
                },
                BodyError::NonFinite,
            ),
            (
                BodyDesc {
                    kind: BodyKind::Dynamic { mass: 0.0 },
                    ..ok
                },
                BodyError::InvalidMass,
            ),
            (
                BodyDesc {
                    kind: BodyKind::Dynamic { mass: -1.0 },
                    ..ok
                },
                BodyError::InvalidMass,
            ),
            (
                BodyDesc {
                    kind: BodyKind::Dynamic { mass: f32::NAN },
                    ..ok
                },
                BodyError::InvalidMass,
            ),
            (
                BodyDesc {
                    orientation: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
                    ..ok
                },
                BodyError::InvalidOrientation,
            ),
        ];
        for (desc, expected) in cases {
            assert_eq!(world.add_body(desc), Err(expected), "{desc:?}");
        }
        assert!(world.is_empty());
    }

    #[test]
    fn stale_handles_never_reach_new_bodies() {
        let mut world = World::default();
        let a = world.add_body(BodyDesc::dynamic(1.0, Vec3::ZERO)).unwrap();
        assert!(world.remove_body(a).is_some());
        assert!(world.remove_body(a).is_none());
        assert!(world.body(a).is_none());

        let b = world.add_body(BodyDesc::dynamic(1.0, Vec3::X)).unwrap();
        assert!(world.body(a).is_none(), "old handle revived by slot reuse");
        assert_eq!(world.body(b).unwrap().position(), Vec3::X);
        assert_eq!(world.len(), 1);
    }

    #[test]
    fn clear_invalidates_all_handles() {
        let mut world = World::default();
        let handles: Vec<_> = (0..3)
            .map(|_| world.add_body(BodyDesc::dynamic(1.0, Vec3::ZERO)).unwrap())
            .collect();
        world.clear();
        assert!(world.is_empty());
        assert!(handles.iter().all(|&h| world.body(h).is_none()));
        world.add_body(BodyDesc::dynamic(1.0, Vec3::ZERO)).unwrap();
        assert!(handles.iter().all(|&h| world.body(h).is_none()));
    }

    #[test]
    fn identical_worlds_stay_bit_identical() {
        let build = || {
            let mut world = World::default();
            for i in 0..10 {
                let desc = BodyDesc {
                    velocity: Vec3::new(i as f32 * 0.3, 2.0, -1.0),
                    ..BodyDesc::dynamic(1.0 + i as f32, Vec3::new(i as f32, 5.0, 0.0))
                };
                world.add_body(desc).unwrap();
            }
            for _ in 0..300 {
                world.step(1.0 / 60.0);
            }
            world
        };
        let bits = |w: &World| -> Vec<[u32; 3]> {
            w.bodies()
                .map(|(_, b)| b.position().to_array().map(f32::to_bits))
                .collect()
        };
        assert_eq!(bits(&build()), bits(&build()));
    }

    #[test]
    fn zero_dt_changes_nothing() {
        let mut world = World::default();
        let h = world.add_body(BodyDesc::dynamic(1.0, Vec3::ONE)).unwrap();
        world.step(0.0);
        assert_eq!(world.body(h).unwrap().position(), Vec3::ONE);
        assert_eq!(world.body(h).unwrap().velocity(), Vec3::ZERO);
    }
}
