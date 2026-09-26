use glam::Vec3;

use crate::body::{BodyDesc, BodyError, RigidBody};
use crate::broadphase::{Aabb, AabbTree};
use crate::contact::Contact;
use crate::contact::contacts_for_pairs;
use crate::solver::{self, ContactCache, SolverSettings};
use crate::union_find::UnionFind;

pub const DEFAULT_GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

/// Below these speeds (m/s, rad/s) for `SLEEP_TIME` seconds, an island may sleep.
const SLEEP_LINEAR_SPEED: f32 = 0.05;
const SLEEP_ANGULAR_SPEED: f32 = 0.1;
const SLEEP_TIME: f32 = 0.5;

/// What one step did and how long each phase took (ms; zero without a clock).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StepStats {
    pub broadphase_ms: f64,
    pub narrowphase_ms: f64,
    /// Waking, the contact solve, and warm-start bookkeeping.
    pub solve_ms: f64,
    /// Integration, sleep updates, and broadphase refits.
    pub integrate_ms: f64,
    pub pairs: usize,
    pub contacts: usize,
    /// Contacts involving an awake body (the ones solved).
    pub active_contacts: usize,
    /// Awake dynamic bodies.
    pub awake: usize,
    pub asleep: usize,
}

impl StepStats {
    pub fn total_ms(&self) -> f64 {
        self.broadphase_ms + self.narrowphase_ms + self.solve_ms + self.integrate_ms
    }
}

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
    /// Broadphase leaf, for bodies with bounded colliders.
    proxy: Option<usize>,
}

/// All bodies plus global settings. Bodies live in a generational arena.
#[derive(Debug, Clone)]
pub struct World {
    gravity: Vec3,
    slots: Vec<Slot>,
    free: Vec<u32>,
    settings: SolverSettings,
    /// Last step's contact impulses, for warm starting.
    cache: ContactCache,
    tree: AabbTree,
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
            settings: SolverSettings::default(),
            cache: ContactCache::default(),
            tree: AabbTree::default(),
        }
    }

    pub fn gravity(&self) -> Vec3 {
        self.gravity
    }

    pub fn solver_settings(&self) -> SolverSettings {
        self.settings
    }

    pub fn set_solver_settings(&mut self, settings: SolverSettings) {
        self.settings = SolverSettings {
            iterations: settings.iterations.max(1),
            ..settings
        };
        if !settings.warm_starting {
            self.cache.clear();
        }
    }

    pub fn add_body(&mut self, desc: BodyDesc) -> Result<BodyHandle, BodyError> {
        let body = RigidBody::new(desc)?;
        let handle = if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.body = Some(body);
            BodyHandle {
                index,
                generation: slot.generation,
            }
        } else {
            let index = u32::try_from(self.slots.len()).expect("more than u32::MAX bodies");
            self.slots.push(Slot {
                generation: 0,
                body: Some(body),
                proxy: None,
            });
            BodyHandle {
                index,
                generation: 0,
            }
        };
        self.slots[handle.index as usize].proxy =
            body_aabb(&body).map(|aabb| self.tree.insert(aabb, handle));
        Ok(handle)
    }

    pub fn remove_body(&mut self, handle: BodyHandle) -> Option<RigidBody> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        let body = slot.body.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        if let Some(proxy) = slot.proxy.take() {
            self.tree.remove(proxy);
        }
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
        self.cache.clear();
        self.tree = AabbTree::default();
        for (index, slot) in self.slots.iter_mut().enumerate() {
            slot.proxy = None;
            if slot.body.take().is_some() {
                slot.generation = slot.generation.wrapping_add(1);
                self.free.push(index as u32);
            }
        }
    }

    /// Advances the simulation by `dt` seconds.
    ///
    /// Order: gravity → find contacts → solve contact velocities (warm-started,
    /// with position error fed back as a velocity bias) → move. Semi-implicit Euler
    /// with the contact solve between the velocity and position updates.
    pub fn step(&mut self, dt: f32) {
        self.step_profiled(dt, &mut || 0.0);
    }

    /// [`World::step`] plus per-phase timings. `clock` returns milliseconds; it is
    /// supplied by the caller because std timers panic on wasm and this crate
    /// depends only on glam.
    pub fn step_profiled(&mut self, dt: f32, clock: &mut dyn FnMut() -> f64) -> StepStats {
        debug_assert!(dt.is_finite() && dt >= 0.0, "invalid timestep {dt}");
        let t0 = clock();
        let pairs = self.candidate_pairs();
        let t1 = clock();
        let contacts = contacts_for_pairs(self, &pairs);
        let t2 = clock();
        let contact_count = contacts.len();
        self.wake_touched_islands(&contacts);
        let gravity = self.gravity;
        for body in self.bodies_mut() {
            body.apply_gravity(gravity, dt);
        }
        // Contacts among sleeping (or static) bodies need no solving.
        let active: Vec<Contact> = contacts
            .into_iter()
            .filter(|c| self.is_active(c.a) || self.is_active(c.b))
            .collect();
        let mut cache = std::mem::take(&mut self.cache);
        solver::solve(self, &active, &mut cache, dt);
        self.cache = cache;
        let t3 = clock();
        for body in self.bodies_mut() {
            body.advance(dt);
        }
        if self.settings.sleeping {
            self.fall_asleep(&active, dt);
        }
        self.sync_broadphase();
        let t4 = clock();

        let (mut awake, mut asleep) = (0, 0);
        for (_, body) in self.bodies() {
            if body.is_asleep() {
                asleep += 1;
            } else if body.inverse_mass() != 0.0 {
                awake += 1;
            }
        }
        StepStats {
            broadphase_ms: t1 - t0,
            narrowphase_ms: t2 - t1,
            solve_ms: t3 - t2,
            integrate_ms: t4 - t3,
            pairs: pairs.len(),
            contacts: contact_count,
            active_contacts: active.len(),
            awake,
            asleep,
        }
    }

    fn is_active(&self, handle: BodyHandle) -> bool {
        self.body(handle).is_some_and(RigidBody::is_active)
    }

    /// Islands: dynamic bodies connected through contacts. Static bodies do not
    /// link islands (the floor would otherwise join everything).
    fn islands(&self, contacts: &[Contact]) -> UnionFind {
        let mut islands = UnionFind::new(self.slots.len());
        for c in contacts {
            let dynamic = |h: BodyHandle| self.body(h).is_some_and(|b| b.inverse_mass() != 0.0);
            if dynamic(c.a) && dynamic(c.b) {
                islands.union(c.a.index as usize, c.b.index as usize);
            }
        }
        islands
    }

    /// Wakes every island that contains an awake dynamic body (e.g. something
    /// landing on a sleeping stack). With sleeping disabled, wakes everything.
    fn wake_touched_islands(&mut self, contacts: &[Contact]) {
        if !self.settings.sleeping {
            for body in self.bodies_mut() {
                body.asleep = false;
            }
            return;
        }
        let mut islands = self.islands(contacts);
        let mut has_awake = vec![false; self.slots.len()];
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.body.as_ref().is_some_and(RigidBody::is_active) {
                has_awake[islands.find(i)] = true;
            }
        }
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if let Some(body) = &mut slot.body
                && body.asleep
                && has_awake[islands.find(i)]
            {
                body.asleep = false;
                body.sleep_timer = 0.0;
            }
        }
    }

    /// Advances sleep timers; an island whose bodies have all rested long enough
    /// goes to sleep together.
    fn fall_asleep(&mut self, contacts: &[Contact], dt: f32) {
        for body in self.bodies_mut() {
            if body.is_active() {
                let resting = body.velocity.length() < SLEEP_LINEAR_SPEED
                    && body.angular_velocity.length() < SLEEP_ANGULAR_SPEED;
                body.sleep_timer = if resting { body.sleep_timer + dt } else { 0.0 };
            }
        }
        let mut islands = self.islands(contacts);
        let mut ready = vec![true; self.slots.len()];
        for (i, slot) in self.slots.iter().enumerate() {
            if let Some(body) = &slot.body
                && body.is_active()
                && body.sleep_timer < SLEEP_TIME
            {
                ready[islands.find(i)] = false;
            }
        }
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if let Some(body) = &mut slot.body
                && body.is_active()
                && ready[islands.find(i)]
            {
                body.asleep = true;
                body.velocity = Vec3::ZERO;
                body.angular_velocity = Vec3::ZERO;
            }
        }
    }

    /// Refits broadphase leaves after bodies moved (cheap when they stay in their margin).
    fn sync_broadphase(&mut self) {
        for slot in &self.slots {
            if let (Some(body), Some(proxy)) = (&slot.body, slot.proxy)
                && let Some(aabb) = body_aabb(body)
            {
                self.tree.update(proxy, aabb);
            }
        }
    }

    /// Pairs that might touch, sorted: tree overlaps between bounded shapes plus every
    /// (plane, bounded dynamic body) pair. Static–static pairs are skipped.
    pub(crate) fn candidate_pairs(&self) -> Vec<(BodyHandle, BodyHandle)> {
        let mut pairs = Vec::new();
        let mut planes = Vec::new();
        for (handle, body) in self.bodies() {
            let Some(collider) = body.collider() else {
                continue;
            };
            if matches!(collider, crate::collider::Collider::Plane { .. }) {
                planes.push(handle);
                continue;
            }
            let proxy = self.slots[handle.index as usize]
                .proxy
                .expect("bounded bodies have proxies");
            let dynamic = body.inverse_mass() != 0.0;
            self.tree.query(self.tree.fat_aabb(proxy), |other| {
                // Each pair once (from its lower handle); skip static–static.
                if other > handle {
                    let other_dynamic = self.body(other).is_some_and(|b| b.inverse_mass() != 0.0);
                    if dynamic || other_dynamic {
                        pairs.push((handle, other));
                    }
                }
            });
        }
        for plane in planes {
            for (handle, body) in self.bodies() {
                if body.inverse_mass() != 0.0 && self.slots[handle.index as usize].proxy.is_some() {
                    pairs.push((plane.min(handle), plane.max(handle)));
                }
            }
        }
        pairs.sort_unstable();
        pairs
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

fn body_aabb(body: &RigidBody) -> Option<Aabb> {
    Aabb::of(body.collider()?, body.position(), body.orientation())
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
