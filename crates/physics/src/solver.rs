//! Sequential-impulse contact solver (Box2D-Lite lineage): accumulated impulse
//! clamping, warm starting from persistent contacts, and Baumgarte position bias.

use std::collections::HashMap;

use glam::{Mat3, Vec3};

use crate::collider::Material;
use crate::contact::Contact;
use crate::world::{BodyHandle, World};

/// Below this approach speed (m/s) contacts don't bounce, so resting bodies don't jitter.
const RESTITUTION_THRESHOLD: f32 = 0.5;
/// Penetration (m) tolerated without position correction, so resting contacts persist.
const SLOP: f32 = 0.005;
/// Fraction of the remaining penetration corrected per step (Baumgarte β).
const BAUMGARTE: f32 = 0.2;
/// Max distance (m, in each body's local frame) for a contact point to count as
/// the same point it was last step.
const MATCH_DISTANCE: f32 = 0.02;

/// Solver knobs, exposed so the demo can show what each one buys.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverSettings {
    /// Velocity iterations per step (≥ 1).
    pub iterations: usize,
    /// Start each step from last step's impulses.
    pub warm_starting: bool,
}

impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            iterations: 10,
            warm_starting: true,
        }
    }
}

/// Last step's impulses for one contact point, keyed by where it sits on each body.
#[derive(Debug, Clone, Copy)]
struct CachedPoint {
    local_a: Vec3,
    local_b: Vec3,
    normal_impulse: f32,
    tangent_impulse: [f32; 2],
}

/// Contact points from the previous step, per body pair.
#[derive(Debug, Clone, Default)]
pub(crate) struct ContactCache {
    pairs: HashMap<(BodyHandle, BodyHandle), Vec<CachedPoint>>,
}

impl ContactCache {
    pub(crate) fn clear(&mut self) {
        self.pairs.clear();
    }
}

/// One side of a contact, frozen for the duration of the solve.
#[derive(Clone, Copy)]
struct Side {
    handle: BodyHandle,
    inv_mass: f32,
    inv_inertia: Mat3,
    /// Contact point relative to the body's center.
    r: Vec3,
}

impl Side {
    fn new(world: &World, handle: BodyHandle, point: Vec3) -> Option<Self> {
        let body = world.body(handle)?;
        Some(Self {
            handle,
            inv_mass: body.inverse_mass(),
            inv_inertia: body.inverse_inertia_world(),
            r: point - body.position(),
        })
    }

    fn velocity(&self, world: &World) -> Vec3 {
        world.body(self.handle).map_or(Vec3::ZERO, |b| {
            b.velocity + b.angular_velocity.cross(self.r)
        })
    }

    fn apply(&self, world: &mut World, impulse: Vec3) {
        if let Some(body) = world.body_mut(self.handle) {
            body.velocity += impulse * self.inv_mass;
            body.angular_velocity += self.inv_inertia * self.r.cross(impulse);
        }
    }

    fn inverse_mass_along(&self, direction: Vec3) -> f32 {
        let angular = (self.inv_inertia * self.r.cross(direction)).cross(self.r);
        self.inv_mass + direction.dot(angular)
    }
}

struct Point {
    pair: (BodyHandle, BodyHandle),
    a: Side,
    b: Side,
    normal: Vec3,
    tangents: [Vec3; 2],
    normal_mass: f32,
    tangent_mass: [f32; 2],
    /// Desired normal separating speed: bounce or position correction.
    target_speed: f32,
    friction: f32,
    normal_impulse: f32,
    tangent_impulse: [f32; 2],
    local_a: Vec3,
    local_b: Vec3,
}

impl Point {
    fn apply_pair(&self, world: &mut World, impulse: Vec3) {
        self.a.apply(world, impulse);
        self.b.apply(world, -impulse);
    }
}

/// Two unit tangents perpendicular to `n` (Catto's construction; deterministic).
fn tangent_basis(n: Vec3) -> [Vec3; 2] {
    let t1 = if n.x.abs() >= 0.577_35 {
        Vec3::new(n.y, -n.x, 0.0)
    } else {
        Vec3::new(0.0, n.z, -n.y)
    }
    .normalize();
    [t1, n.cross(t1)]
}

fn local_point(world: &World, handle: BodyHandle, point: Vec3) -> Option<Vec3> {
    let body = world.body(handle)?;
    Some(body.orientation().inverse() * (point - body.position()))
}

/// Resolves all contacts' velocities for one step and refreshes `cache`.
pub(crate) fn solve(world: &mut World, contacts: &[Contact], cache: &mut ContactCache, dt: f32) {
    let settings = world.solver_settings();
    let mut previous = std::mem::take(&mut cache.pairs);
    let mut points: Vec<Point> = Vec::with_capacity(contacts.len());

    for c in contacts {
        let (Some(a), Some(b)) = (
            Side::new(world, c.a, c.point),
            Side::new(world, c.b, c.point),
        ) else {
            continue;
        };
        let normal_inv_mass = a.inverse_mass_along(c.normal) + b.inverse_mass_along(c.normal);
        if normal_inv_mass <= 0.0 {
            continue;
        }
        let (Some(local_a), Some(local_b)) = (
            local_point(world, c.a, c.point),
            local_point(world, c.b, c.point),
        ) else {
            continue;
        };
        let (Some(body_a), Some(body_b)) = (world.body(c.a), world.body(c.b)) else {
            continue;
        };
        let (restitution, friction) = Material::combine(body_a.material(), body_b.material());
        let tangents = tangent_basis(c.normal);

        // Persistent contact: take over last step's impulses if this is the same point.
        let mut normal_impulse = 0.0;
        let mut tangent_impulse = [0.0; 2];
        if settings.warm_starting
            && let Some(old) = previous.get_mut(&(c.a, c.b))
            && let Some(i) = old.iter().position(|o| {
                o.local_a.distance(local_a) < MATCH_DISTANCE
                    && o.local_b.distance(local_b) < MATCH_DISTANCE
            })
        {
            let matched = old.swap_remove(i); // each old point matches at most once
            normal_impulse = matched.normal_impulse;
            tangent_impulse = matched.tangent_impulse;
        }

        let approach = (a.velocity(world) - b.velocity(world)).dot(c.normal);
        let bounce = if approach < -RESTITUTION_THRESHOLD {
            -restitution * approach
        } else {
            0.0
        };
        let correction = BAUMGARTE / dt * (c.depth - SLOP).max(0.0);
        points.push(Point {
            pair: (c.a, c.b),
            a,
            b,
            normal: c.normal,
            tangents,
            normal_mass: 1.0 / normal_inv_mass,
            tangent_mass: tangents
                .map(|t| 1.0 / (a.inverse_mass_along(t) + b.inverse_mass_along(t))),
            target_speed: bounce.max(correction),
            friction,
            normal_impulse,
            tangent_impulse,
            local_a,
            local_b,
        });
    }

    // Warm start: re-apply last step's impulses before iterating.
    for p in &points {
        let impulse = p.normal * p.normal_impulse
            + p.tangents[0] * p.tangent_impulse[0]
            + p.tangents[1] * p.tangent_impulse[1];
        if impulse != Vec3::ZERO {
            p.apply_pair(world, impulse);
        }
    }

    // Symmetric Gauss–Seidel: alternate sweep direction so the fixed visiting order
    // doesn't bias redundant contacts (4 coplanar points per face) toward one side.
    // A one-directional sweep leaves a same-signed residual spin every step that
    // accumulates into a lean in tall stacks.
    for iteration in 0..settings.iterations.max(1) {
        if iteration % 2 == 0 {
            points.iter_mut().for_each(|p| solve_point(world, p));
        } else {
            points.iter_mut().rev().for_each(|p| solve_point(world, p));
        }
    }

    for p in &points {
        cache.pairs.entry(p.pair).or_default().push(CachedPoint {
            local_a: p.local_a,
            local_b: p.local_b,
            normal_impulse: p.normal_impulse,
            tangent_impulse: p.tangent_impulse,
        });
    }
}

/// One Gauss–Seidel update of a contact point: friction, then the normal impulse.
fn solve_point(world: &mut World, p: &mut Point) {
    // Friction first (Box2D order), bounded by the current normal impulse.
    let max_friction = p.friction * p.normal_impulse;
    for k in 0..2 {
        let relative = p.a.velocity(world) - p.b.velocity(world);
        let delta = -relative.dot(p.tangents[k]) * p.tangent_mass[k];
        let total = (p.tangent_impulse[k] + delta).clamp(-max_friction, max_friction);
        let applied = total - p.tangent_impulse[k];
        p.tangent_impulse[k] = total;
        p.apply_pair(world, p.tangents[k] * applied);
    }

    // Accumulated clamp: the step's *total* push stays ≥ 0, so a later
    // iteration can take back an earlier over-push.
    let relative = p.a.velocity(world) - p.b.velocity(world);
    let delta = (p.target_speed - relative.dot(p.normal)) * p.normal_mass;
    let total = (p.normal_impulse + delta).max(0.0);
    let applied = total - p.normal_impulse;
    p.normal_impulse = total;
    p.apply_pair(world, p.normal * applied);
}
