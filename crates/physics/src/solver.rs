//! Sequential-impulse contact solver (Box2D lineage) with linear and angular terms.

use std::collections::HashMap;

use glam::{Mat3, Vec3};

use crate::collider::Material;
use crate::contact::{Contact, find_contacts};
use crate::world::{BodyHandle, World};

const ITERATIONS: usize = 8;
/// Below this approach speed (m/s) contacts don't bounce, so resting bodies don't jitter.
const RESTITUTION_THRESHOLD: f32 = 0.5;
/// Penetration (m) tolerated without correction, so resting contacts persist.
const SLOP: f32 = 0.01;
/// Fraction of the remaining penetration removed per step.
const CORRECTION: f32 = 0.8;

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

    /// Velocity of the contact point on this body.
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

    /// This side's share of `1 / effective mass` along `direction`.
    fn inverse_mass_along(&self, direction: Vec3) -> f32 {
        let angular = (self.inv_inertia * self.r.cross(direction)).cross(self.r);
        self.inv_mass + direction.dot(angular)
    }
}

struct Constraint {
    a: Side,
    b: Side,
    normal: Vec3,
    normal_inv_mass: f32,
    /// Desired normal separating speed after the solve.
    target_speed: f32,
    friction: f32,
}

/// Adjusts velocities so contacts stop approaching (bouncing by restitution) and
/// friction opposes sliding at the contact point — which makes spheres roll.
pub(crate) fn solve_velocities(world: &mut World, contacts: &[Contact]) {
    let constraints: Vec<Constraint> = contacts
        .iter()
        .filter_map(|c| {
            let (a, b) = (
                Side::new(world, c.a, c.point)?,
                Side::new(world, c.b, c.point)?,
            );
            let normal_inv_mass = a.inverse_mass_along(c.normal) + b.inverse_mass_along(c.normal);
            if normal_inv_mass <= 0.0 {
                return None;
            }
            let (ma, mb) = (world.body(c.a)?.material(), world.body(c.b)?.material());
            let (restitution, friction) = Material::combine(ma, mb);
            let approach = (a.velocity(world) - b.velocity(world)).dot(c.normal);
            let target_speed = if approach < -RESTITUTION_THRESHOLD {
                -restitution * approach
            } else {
                0.0
            };
            Some(Constraint {
                a,
                b,
                normal: c.normal,
                normal_inv_mass,
                target_speed,
                friction,
            })
        })
        .collect();

    for _ in 0..ITERATIONS {
        for c in &constraints {
            let relative = c.a.velocity(world) - c.b.velocity(world);
            // Per-iteration clamp: a contact can push, never pull. M4 replaces this
            // with accumulated clamping so later iterations can undo over-pushing.
            let jn = ((c.target_speed - relative.dot(c.normal)) / c.normal_inv_mass).max(0.0);
            c.a.apply(world, c.normal * jn);
            c.b.apply(world, -c.normal * jn);

            let relative = c.a.velocity(world) - c.b.velocity(world);
            let tangential = relative - c.normal * relative.dot(c.normal);
            let slide_speed = tangential.length();
            if slide_speed > 1e-6 {
                let direction = tangential / slide_speed;
                let tangent_inv_mass =
                    c.a.inverse_mass_along(direction) + c.b.inverse_mass_along(direction);
                // Coulomb: friction can't exceed μ times this iteration's normal impulse.
                let jt = (slide_speed / tangent_inv_mass).min(c.friction * jn);
                c.a.apply(world, -direction * jt);
                c.b.apply(world, direction * jt);
            }
        }
    }
}

/// Pushes overlapping bodies apart after integration (no velocity change). Uses the
/// deepest point per body pair: a multi-point manifold would otherwise push N times.
pub(crate) fn correct_positions(world: &mut World) {
    let mut deepest: HashMap<(BodyHandle, BodyHandle), Contact> = HashMap::new();
    for c in find_contacts(world) {
        deepest
            .entry((c.a, c.b))
            .and_modify(|d| {
                if c.depth > d.depth {
                    *d = c;
                }
            })
            .or_insert(c);
    }
    // Deterministic order (HashMap iteration order is randomized per process).
    let mut pairs: Vec<Contact> = deepest.into_values().collect();
    pairs.sort_by_key(|c| (c.a, c.b));
    for c in pairs {
        let (Some(a), Some(b)) = (world.body(c.a), world.body(c.b)) else {
            continue;
        };
        let (inv_a, inv_b) = (a.inverse_mass(), b.inverse_mass());
        let inv_sum = inv_a + inv_b;
        if inv_sum == 0.0 {
            continue;
        }
        let push = c.normal * ((c.depth - SLOP).max(0.0) * CORRECTION / inv_sum);
        if let Some(a) = world.body_mut(c.a) {
            a.position += push * inv_a;
        }
        if let Some(b) = world.body_mut(c.b) {
            b.position -= push * inv_b;
        }
    }
}
