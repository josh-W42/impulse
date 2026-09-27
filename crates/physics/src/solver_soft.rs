//! Soft step solver (M11), after Box2D v3's: each step is split into substeps
//! that integrate velocities, warm start, solve soft joints and contacts once,
//! integrate positions, and relax (solve again without position correction,
//! removing the push-apart velocity); a final pass applies restitution.
//!
//! Contacts are found once per step. Within it, a contact's separation is
//! tracked from how far its two anchor points (fixed on their bodies) move
//! apart along the normal, so there is no re-collision per substep.

use std::collections::HashMap;

use glam::Vec3;

use crate::collider::Material;
use crate::contact::Contact;
use crate::joint::{self, Joint, Soft};
use crate::solver::{CachedPoint, ContactCache, MATCH_DISTANCE, Side, local_point, tangent_basis};
use crate::world::{BodyHandle, World};

/// Contact spring stiffness and damping ratio: Box2D v3's defaults
/// (`b2DefaultWorldDef`). Stiffness is capped at a quarter of the substep rate.
const CONTACT_HERTZ: f32 = 30.0;
const CONTACT_DAMPING: f32 = 10.0;
/// Fastest an overlap is pushed apart (m/s), so deep overlaps don't explode.
const MAX_PUSH: f32 = 3.0;
/// Approach speed (m/s) below which contacts don't bounce.
const RESTITUTION_THRESHOLD: f32 = 1.0;

struct SoftContact {
    pair: (BodyHandle, BodyHandle),
    /// The contact point in each body's frame (equal in the world at prepare);
    /// rotates with the bodies, for tracking separation.
    local_a: Vec3,
    local_b: Vec3,
    /// Lever arms (point − center) at prepare, in world orientation. Impulses
    /// and velocities use these fixed arms, as Box2D v3 does: arms rotating
    /// with the bodies would misplace friction on a rolling ball.
    arm_a: Vec3,
    arm_b: Vec3,
    /// From b toward a.
    normal: Vec3,
    tangents: [Vec3; 2],
    /// Separation at prepare (negative: overlapping).
    separation: f32,
    /// Normal velocity (a relative to b) before solving, for restitution.
    approach: f32,
    restitution: f32,
    friction: f32,
    normal_impulse: f32,
    tangent_impulse: [f32; 2],
    /// Largest normal impulse applied in any pass (restitution only where the
    /// contact actually pushed).
    max_normal_impulse: f32,
}

/// A contact's geometry at the bodies' current poses.
struct Frame {
    a: Side,
    b: Side,
    separation: f32,
}

impl SoftContact {
    fn frame(&self, world: &World) -> Option<Frame> {
        let (body_a, body_b) = (world.body(self.pair.0)?, world.body(self.pair.1)?);
        let pa = body_a.position() + body_a.orientation() * self.local_a;
        let pb = body_b.position() + body_b.orientation() * self.local_b;
        Some(Frame {
            a: Side::new(world, self.pair.0, body_a.position() + self.arm_a)?,
            b: Side::new(world, self.pair.1, body_b.position() + self.arm_b)?,
            separation: self.separation + (pa - pb).dot(self.normal),
        })
    }
}

fn push(world: &mut World, frame: &Frame, impulse: Vec3) {
    frame.a.apply(world, impulse);
    frame.b.apply(world, -impulse);
}

fn relative_velocity(world: &World, frame: &Frame) -> Vec3 {
    frame.a.velocity(world) - frame.b.velocity(world)
}

/// Builds this step's contacts, taking over last step's impulses for points
/// that persist (when warm starting).
fn prepare(
    world: &World,
    contacts: &[Contact],
    previous: &mut HashMap<(BodyHandle, BodyHandle), Vec<CachedPoint>>,
    warm: bool,
) -> Vec<SoftContact> {
    let mut out = Vec::with_capacity(contacts.len());
    for c in contacts {
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
        let (mut normal_impulse, mut tangent_impulse) = (0.0, [0.0; 2]);
        if warm
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
        let mut contact = SoftContact {
            pair: (c.a, c.b),
            local_a,
            local_b,
            arm_a: c.point - body_a.position(),
            arm_b: c.point - body_b.position(),
            normal: c.normal,
            tangents: tangent_basis(c.normal),
            separation: -c.depth,
            approach: 0.0,
            restitution,
            friction,
            normal_impulse,
            tangent_impulse,
            max_normal_impulse: 0.0,
        };
        if let Some(frame) = contact.frame(world) {
            contact.approach = relative_velocity(world, &frame).dot(c.normal);
            out.push(contact);
        }
    }
    out
}

fn warm_start(world: &mut World, c: &SoftContact) {
    let impulse = c.normal * c.normal_impulse
        + c.tangents[0] * c.tangent_impulse[0]
        + c.tangents[1] * c.tangent_impulse[1];
    if impulse != Vec3::ZERO
        && let Some(frame) = c.frame(world)
    {
        push(world, &frame, impulse);
    }
}

/// One pass over a contact: the soft normal row (with the push-apart bias,
/// or without it when relaxing), then friction bounded by the normal impulse.
fn solve(world: &mut World, c: &mut SoftContact, soft: &Soft, use_bias: bool, inv_h: f32) {
    let Some(frame) = c.frame(world) else {
        return;
    };
    let n = c.normal;
    let k = frame.a.inverse_mass_along(n) + frame.b.inverse_mass_along(n);
    if k > 0.0 {
        let vn = relative_velocity(world, &frame).dot(n);
        // No slop: the spring's steady overlap keeps resting contacts alive,
        // and a dead zone would let tall stacks rock (see M11 results).
        let s = frame.separation;
        let (bias, mass_scale, impulse_scale) = if s > 0.0 {
            // Speculative: may close this substep's gap, and no more.
            (s * inv_h, 1.0, 0.0)
        } else if use_bias {
            (
                (soft.bias_rate * s).max(-MAX_PUSH),
                soft.mass_scale,
                soft.impulse_scale,
            )
        } else {
            (0.0, 1.0, 0.0)
        };
        let impulse = -mass_scale * (vn + bias) / k - impulse_scale * c.normal_impulse;
        let total = (c.normal_impulse + impulse).max(0.0);
        let applied = total - c.normal_impulse;
        c.normal_impulse = total;
        c.max_normal_impulse = c.max_normal_impulse.max(applied);
        push(world, &frame, n * applied);
    }
    let max_friction = c.friction * c.normal_impulse;
    for i in 0..2 {
        let t = c.tangents[i];
        let kt = frame.a.inverse_mass_along(t) + frame.b.inverse_mass_along(t);
        if kt <= 0.0 {
            continue;
        }
        let vt = relative_velocity(world, &frame).dot(t);
        let total = (c.tangent_impulse[i] - vt / kt).clamp(-max_friction, max_friction);
        let applied = total - c.tangent_impulse[i];
        c.tangent_impulse[i] = total;
        push(world, &frame, t * applied);
    }
}

/// Bounce, from the approach speed measured before solving.
fn restitution(world: &mut World, c: &mut SoftContact) {
    if c.restitution == 0.0 || c.approach > -RESTITUTION_THRESHOLD || c.max_normal_impulse == 0.0 {
        return;
    }
    let Some(frame) = c.frame(world) else {
        return;
    };
    let n = c.normal;
    let k = frame.a.inverse_mass_along(n) + frame.b.inverse_mass_along(n);
    if k <= 0.0 {
        return;
    }
    let vn = relative_velocity(world, &frame).dot(n);
    let impulse = -(vn + c.restitution * c.approach) / k;
    let total = (c.normal_impulse + impulse).max(0.0);
    let applied = total - c.normal_impulse;
    c.normal_impulse = total;
    push(world, &frame, n * applied);
}

/// Advances the world by `dt`: substeps, relax, restitution. Gravity and
/// position integration happen here (per substep), not in the caller.
pub(crate) fn step(
    world: &mut World,
    contacts: &[Contact],
    cache: &mut ContactCache,
    joints: &mut [Option<Joint>],
    dt: f32,
) {
    let settings = world.solver_settings();
    let substeps = settings.substeps.max(1);
    let h = dt / substeps as f32;
    if h <= 0.0 {
        return;
    }
    let inv_h = 1.0 / h;
    let soft = Soft::new(CONTACT_HERTZ.min(0.25 * inv_h), CONTACT_DAMPING, h);
    let mut previous = std::mem::take(&mut cache.pairs);
    let mut points = prepare(world, contacts, &mut previous, settings.warm_starting);

    for _ in 0..substeps {
        world.integrate_velocities(h);
        // Joints are re-prepared from the current poses each substep.
        let mut rows = joint::prepare(world, joints, h, settings.warm_starting);
        for r in &mut rows {
            r.warm_start(world);
        }
        for c in &points {
            warm_start(world, c);
        }
        for r in &mut rows {
            r.solve(world, true);
        }
        for c in &mut points {
            solve(world, c, &soft, true, inv_h);
        }
        world.integrate_positions(h);
        // Relax: solve again without position correction, so the push-apart
        // velocity that just moved the bodies isn't kept as momentum into the
        // next substep (Box2D v3 relaxes every substep).
        for r in &mut rows {
            r.solve(world, false);
        }
        for c in &mut points {
            solve(world, c, &soft, false, inv_h);
        }
        for r in &rows {
            r.store(joints);
        }
    }
    for c in &mut points {
        restitution(world, c);
    }

    for c in &points {
        cache.pairs.entry(c.pair).or_default().push(CachedPoint {
            local_a: c.local_a,
            local_b: c.local_b,
            normal_impulse: c.normal_impulse,
            tangent_impulse: c.tangent_impulse,
        });
    }
}
