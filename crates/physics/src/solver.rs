//! Sequential-impulse contact solver (Box2D lineage), linear only until M3.

use glam::Vec3;

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

struct Constraint {
    a: BodyHandle,
    b: BodyHandle,
    normal: Vec3,
    inv_mass_a: f32,
    inv_mass_b: f32,
    /// Desired normal separating speed after the solve.
    target_speed: f32,
    friction: f32,
}

fn velocity(world: &World, handle: BodyHandle) -> Vec3 {
    world.body(handle).map_or(Vec3::ZERO, |b| b.velocity)
}

fn apply_impulse(world: &mut World, handle: BodyHandle, impulse: Vec3) {
    if let Some(body) = world.body_mut(handle) {
        body.velocity += impulse * body.inverse_mass();
    }
}

/// Adjusts velocities so contacts stop approaching (bouncing by restitution) and
/// friction opposes sliding.
pub(crate) fn solve_velocities(world: &mut World, contacts: &[Contact]) {
    let constraints: Vec<Constraint> = contacts
        .iter()
        .filter_map(|c| {
            let (a, b) = (world.body(c.a)?, world.body(c.b)?);
            let (restitution, friction) = Material::combine(a.material(), b.material());
            let approach = (a.velocity - b.velocity).dot(c.normal);
            let target_speed = if approach < -RESTITUTION_THRESHOLD {
                -restitution * approach
            } else {
                0.0
            };
            Some(Constraint {
                a: c.a,
                b: c.b,
                normal: c.normal,
                inv_mass_a: a.inverse_mass(),
                inv_mass_b: b.inverse_mass(),
                target_speed,
                friction,
            })
        })
        .filter(|c| c.inv_mass_a + c.inv_mass_b > 0.0)
        .collect();

    for _ in 0..ITERATIONS {
        for c in &constraints {
            let inv_sum = c.inv_mass_a + c.inv_mass_b;
            let relative = velocity(world, c.a) - velocity(world, c.b);
            let normal_speed = relative.dot(c.normal);
            // Per-iteration clamp: a contact can push, never pull. M4 replaces this
            // with accumulated clamping so later iterations can undo over-pushing.
            let jn = ((c.target_speed - normal_speed) / inv_sum).max(0.0);
            apply_impulse(world, c.a, c.normal * jn);
            apply_impulse(world, c.b, -c.normal * jn);

            let relative = velocity(world, c.a) - velocity(world, c.b);
            let tangential = relative - c.normal * relative.dot(c.normal);
            let slide_speed = tangential.length();
            if slide_speed > 1e-6 {
                // Coulomb: friction can't exceed μ times this iteration's normal impulse.
                let jt = (slide_speed / inv_sum).min(c.friction * jn);
                let direction = tangential / slide_speed;
                apply_impulse(world, c.a, -direction * jt);
                apply_impulse(world, c.b, direction * jt);
            }
        }
    }
}

/// Pushes overlapping bodies apart after integration (no velocity change).
pub(crate) fn correct_positions(world: &mut World) {
    for c in find_contacts(world) {
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
