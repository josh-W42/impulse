//! Joints: velocity constraints between two bodies, solved alongside contacts by
//! the sequential-impulse solver with the same machinery (accumulated impulses,
//! warm starting). Equality rows are *soft* (a stiff, critically damped spring;
//! Box2D v3's soft step): the compliance keeps redundant rows, as in any closed
//! loop of hinges, from drifting into ever-larger impulses that fight each other.

use std::f32::consts::{PI, TAU};
use std::fmt;

use glam::{Mat2, Mat3, Quat, Vec2, Vec3};

use crate::solver::{Side, tangent_basis};
use crate::world::{BodyHandle, World};

/// Fraction of a limit's overshoot corrected per step.
const BAUMGARTE: f32 = 0.2;
/// Stiffness (Hz) and damping ratio of the soft equality rows.
const JOINT_HERTZ: f32 = 30.0;
const JOINT_DAMPING: f32 = 1.0;
/// Limits nearer than this (rad) join the solve as speculative rows, so fast
/// swings stop at the limit instead of overshooting and bouncing back.
const LIMIT_MARGIN: f32 = 0.25;
/// Shortest allowed distance-joint length (m); shorter should be a ball-socket.
const MIN_LENGTH: f32 = 1e-3;

/// Refers to a joint in a [`World`]. Ids are not reused until `World::clear`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JointId(pub(crate) u32);

/// Caps how far a ball-socket may swing: the angle between `axis` as carried by
/// each body (equal when the joint is made) stays at most `max_angle`. Twist
/// about the axis is free.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cone {
    pub axis: Vec3,
    pub max_angle: f32,
}

/// What a joint holds together. Points and axes are world-space, taken at the
/// bodies' current poses when the joint is added.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointKind {
    /// The bodies share `pivot`; rotation is free (within the optional cone).
    BallSocket { pivot: Vec3, cone: Option<Cone> },
    /// The bodies share `pivot` and turn only about `axis`. `limits` bound the
    /// angle (rad, right-handed about `axis`) of `b` relative to `a`, zero at
    /// creation, within [-π, π].
    Hinge {
        pivot: Vec3,
        axis: Vec3,
        limits: Option<(f32, f32)>,
    },
    /// The anchors keep their current distance: a massless rigid rod.
    Distance { anchor_a: Vec3, anchor_b: Vec3 },
}

/// Everything needed to add a joint. `collide_connected` (default false) lets the
/// two bodies still collide with each other.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointDesc {
    pub a: BodyHandle,
    pub b: BodyHandle,
    pub kind: JointKind,
    pub collide_connected: bool,
}

impl JointDesc {
    pub fn ball_socket(a: BodyHandle, b: BodyHandle, pivot: Vec3) -> Self {
        Self::new(a, b, JointKind::BallSocket { pivot, cone: None })
    }

    pub fn hinge(a: BodyHandle, b: BodyHandle, pivot: Vec3, axis: Vec3) -> Self {
        let kind = JointKind::Hinge {
            pivot,
            axis,
            limits: None,
        };
        Self::new(a, b, kind)
    }

    pub fn distance(a: BodyHandle, b: BodyHandle, anchor_a: Vec3, anchor_b: Vec3) -> Self {
        Self::new(a, b, JointKind::Distance { anchor_a, anchor_b })
    }

    fn new(a: BodyHandle, b: BodyHandle, kind: JointKind) -> Self {
        Self {
            a,
            b,
            kind,
            collide_connected: false,
        }
    }

    /// Adds a swing limit. Ball-sockets only (debug builds panic otherwise).
    pub fn with_cone(mut self, axis: Vec3, max_angle: f32) -> Self {
        match &mut self.kind {
            JointKind::BallSocket { cone, .. } => *cone = Some(Cone { axis, max_angle }),
            _ => debug_assert!(false, "cone limits apply to ball-socket joints"),
        }
        self
    }

    /// Adds angle limits. Hinges only (debug builds panic otherwise).
    pub fn with_limits(mut self, min: f32, max: f32) -> Self {
        match &mut self.kind {
            JointKind::Hinge { limits, .. } => *limits = Some((min, max)),
            _ => debug_assert!(false, "angle limits apply to hinge joints"),
        }
        self
    }

    pub fn with_collide_connected(mut self, collide: bool) -> Self {
        self.collide_connected = collide;
        self
    }
}

/// Why a joint could not be added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointError {
    SameBody,
    UnknownBody,
    NoDynamicBody,
    NonFinite,
    ZeroAxis,
    InvalidLimits,
    InvalidCone,
    CoincidentAnchors,
}

impl fmt::Display for JointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SameBody => write!(f, "a joint needs two different bodies"),
            Self::UnknownBody => write!(f, "a joint body does not exist"),
            Self::NoDynamicBody => write!(f, "at least one joint body must be dynamic"),
            Self::NonFinite => write!(f, "joint points and axes must be finite"),
            Self::ZeroAxis => write!(f, "joint axes must be non-zero"),
            Self::InvalidLimits => write!(f, "hinge limits need -π <= min <= max <= π"),
            Self::InvalidCone => write!(f, "cone angles must lie strictly between 0 and π"),
            Self::CoincidentAnchors => write!(f, "distance-joint anchors must be apart"),
        }
    }
}

impl std::error::Error for JointError {}

/// Last step's total impulses, for warm starting.
#[derive(Debug, Clone, Copy, Default)]
struct Accumulated {
    point: Vec3,
    /// Hinge axis-alignment torque impulse (world space).
    angular: Vec3,
    lower: f32,
    upper: f32,
    distance: f32,
}

#[derive(Debug, Clone, Copy)]
enum Constraint {
    BallSocket {
        /// Cone axis in each body's frame, and the limit.
        cone: Option<(Vec3, Vec3, f32)>,
    },
    Hinge {
        axis_a: Vec3,
        axis_b: Vec3,
        /// A perpendicular to the axis, per body, for measuring the angle.
        ref_a: Vec3,
        ref_b: Vec3,
        limits: Option<(f32, f32)>,
    },
    Distance {
        length: f32,
    },
}

/// A validated joint, with anchors and axes stored in each body's frame.
#[derive(Debug, Clone)]
pub(crate) struct Joint {
    a: BodyHandle,
    b: BodyHandle,
    collide_connected: bool,
    local_a: Vec3,
    local_b: Vec3,
    constraint: Constraint,
    impulse: Accumulated,
}

fn finite(v: Vec3) -> Result<Vec3, JointError> {
    v.is_finite().then_some(v).ok_or(JointError::NonFinite)
}

fn unit(v: Vec3) -> Result<Vec3, JointError> {
    finite(v)?.try_normalize().ok_or(JointError::ZeroAxis)
}

impl Joint {
    pub(crate) fn new(desc: &JointDesc, world: &World) -> Result<Self, JointError> {
        if desc.a == desc.b {
            return Err(JointError::SameBody);
        }
        let (Some(a), Some(b)) = (world.body(desc.a), world.body(desc.b)) else {
            return Err(JointError::UnknownBody);
        };
        if a.inverse_mass() == 0.0 && b.inverse_mass() == 0.0 {
            return Err(JointError::NoDynamicBody);
        }
        let (inv_a, inv_b) = (a.orientation().inverse(), b.orientation().inverse());
        let point_a = |p: Vec3| inv_a * (p - a.position());
        let point_b = |p: Vec3| inv_b * (p - b.position());
        let (local_a, local_b, constraint) = match desc.kind {
            JointKind::BallSocket { pivot, cone } => {
                let pivot = finite(pivot)?;
                let cone = match cone {
                    Some(Cone { axis, max_angle }) => {
                        let axis = unit(axis)?;
                        if !(max_angle > 0.0 && max_angle < PI) {
                            return Err(JointError::InvalidCone);
                        }
                        Some((inv_a * axis, inv_b * axis, max_angle))
                    }
                    None => None,
                };
                (
                    point_a(pivot),
                    point_b(pivot),
                    Constraint::BallSocket { cone },
                )
            }
            JointKind::Hinge {
                pivot,
                axis,
                limits,
            } => {
                let pivot = finite(pivot)?;
                let axis = unit(axis)?;
                if let Some((min, max)) = limits
                    && !(-PI <= min && min <= max && max <= PI)
                {
                    return Err(JointError::InvalidLimits);
                }
                let reference = tangent_basis(axis)[0];
                let constraint = Constraint::Hinge {
                    axis_a: inv_a * axis,
                    axis_b: inv_b * axis,
                    ref_a: inv_a * reference,
                    ref_b: inv_b * reference,
                    limits,
                };
                (point_a(pivot), point_b(pivot), constraint)
            }
            JointKind::Distance { anchor_a, anchor_b } => {
                let (anchor_a, anchor_b) = (finite(anchor_a)?, finite(anchor_b)?);
                let length = anchor_a.distance(anchor_b);
                if length < MIN_LENGTH {
                    return Err(JointError::CoincidentAnchors);
                }
                (
                    point_a(anchor_a),
                    point_b(anchor_b),
                    Constraint::Distance { length },
                )
            }
        };
        Ok(Self {
            a: desc.a,
            b: desc.b,
            collide_connected: desc.collide_connected,
            local_a,
            local_b,
            constraint,
            impulse: Accumulated::default(),
        })
    }

    pub(crate) fn bodies(&self) -> (BodyHandle, BodyHandle) {
        (self.a, self.b)
    }

    pub(crate) fn collide_connected(&self) -> bool {
        self.collide_connected
    }

    /// World-space anchors on each body; `None` once either body is gone.
    pub(crate) fn anchors(&self, world: &World) -> Option<(Vec3, Vec3)> {
        let (a, b) = (world.body(self.a)?, world.body(self.b)?);
        Some((
            a.position() + a.orientation() * self.local_a,
            b.position() + b.orientation() * self.local_b,
        ))
    }
}

/// `v × u` as a matrix product.
fn skew(v: Vec3) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(0.0, v.z, -v.y),
        Vec3::new(-v.z, 0.0, v.x),
        Vec3::new(v.y, -v.x, 0.0),
    )
}

/// One side's contribution to the point-constraint mass: `m⁻¹·𝟙 − [r]ₓ I⁻¹ [r]ₓ`.
fn point_mass(side: &Side) -> Mat3 {
    let r = skew(side.r);
    Mat3::from_diagonal(Vec3::splat(side.inv_mass)) - r * side.inv_inertia * r
}

/// Keeps the anchors together: all three directions at once (3×3 block).
struct PointRow {
    inv_k: Mat3,
    bias: Vec3,
}

/// Keeps the hinge axes aligned: the two directions perpendicular to the axis.
struct AlignRow {
    dirs: [Vec3; 2],
    inv_k: Mat2,
    bias: Vec2,
}

/// One-sided angular row: `(ω_a − ω_b)·dir ≥ bias`, impulse ≥ 0.
struct LimitRow {
    dir: Vec3,
    mass: f32,
    bias: f32,
}

/// Keeps the anchors `length` apart along `normal` (a → b's anchor direction).
struct LengthRow {
    normal: Vec3,
    mass: f32,
    bias: f32,
}

/// Soft-constraint coefficients for one step: the row solves
/// `λ = −mass_scale·m·(Ċ + bias_rate·C) − impulse_scale·λ_total`.
#[derive(Clone, Copy)]
struct Soft {
    bias_rate: f32,
    mass_scale: f32,
    impulse_scale: f32,
}

impl Soft {
    fn new(hertz: f32, damping: f32, dt: f32) -> Self {
        let omega = TAU * hertz;
        let a1 = 2.0 * damping + dt * omega;
        let a2 = dt * omega * a1;
        let a3 = 1.0 / (1.0 + a2);
        Self {
            bias_rate: omega / a1,
            mass_scale: a2 * a3,
            impulse_scale: a3,
        }
    }
}

/// A joint's rows for one step, with geometry frozen at the start of the step.
pub(crate) struct Rows {
    index: usize,
    soft: Soft,
    a: Side,
    b: Side,
    impulse: Accumulated,
    point: Option<PointRow>,
    align: Option<AlignRow>,
    lower: Option<LimitRow>,
    upper: Option<LimitRow>,
    length: Option<LengthRow>,
}

fn limit_row(a: &Side, b: &Side, dir: Vec3, room: f32, dt: f32) -> Option<LimitRow> {
    let inv = dir.dot((a.inv_inertia + b.inv_inertia) * dir);
    // Past the limit: push back (Baumgarte). Short of it: allow closing the gap
    // this step and no more (speculative).
    let bias = if room < 0.0 {
        -BAUMGARTE / dt * room
    } else {
        -room / dt
    };
    (inv > 0.0).then(|| LimitRow {
        dir,
        mass: 1.0 / inv,
        bias,
    })
}

/// Builds this step's rows for every joint touching an awake dynamic body.
pub(crate) fn prepare(world: &World, joints: &[Option<Joint>], dt: f32, warm: bool) -> Vec<Rows> {
    let soft = Soft::new(JOINT_HERTZ, JOINT_DAMPING, dt);
    let correction = soft.bias_rate;
    let mut out = Vec::new();
    for (index, joint) in joints.iter().enumerate() {
        let Some(j) = joint else { continue };
        let (Some(body_a), Some(body_b)) = (world.body(j.a), world.body(j.b)) else {
            continue;
        };
        if !(body_a.is_active() || body_b.is_active()) {
            continue;
        }
        let (qa, qb): (Quat, Quat) = (body_a.orientation(), body_b.orientation());
        let pa = body_a.position() + qa * j.local_a;
        let pb = body_b.position() + qb * j.local_b;
        let (Some(a), Some(b)) = (Side::new(world, j.a, pa), Side::new(world, j.b, pb)) else {
            continue;
        };
        let mut rows = Rows {
            index,
            soft,
            a,
            b,
            impulse: if warm {
                j.impulse
            } else {
                Accumulated::default()
            },
            point: None,
            align: None,
            lower: None,
            upper: None,
            length: None,
        };
        let point_row = || {
            let k = point_mass(&a) + point_mass(&b);
            (k.determinant().abs() > 1e-12).then(|| PointRow {
                inv_k: k.inverse(),
                bias: correction * (pa - pb),
            })
        };
        match j.constraint {
            Constraint::BallSocket { cone } => {
                rows.point = point_row();
                if let Some((axis_a, axis_b, max_angle)) = cone {
                    let (wa, wb) = (qa * axis_a, qb * axis_b);
                    let room = max_angle - wa.angle_between(wb);
                    // The swing grows at (ω_b − ω_a)·dir.
                    if room < LIMIT_MARGIN
                        && let Some(dir) = wa.cross(wb).try_normalize()
                    {
                        rows.upper = limit_row(&a, &b, dir, room, dt);
                    }
                }
            }
            Constraint::Hinge {
                axis_a,
                axis_b,
                ref_a,
                ref_b,
                limits,
            } => {
                rows.point = point_row();
                let (wa, wb) = (qa * axis_a, qb * axis_b);
                let dirs = tangent_basis(wa);
                // b's axis is rotated from a's by ≈ wa × wb; (ω_a − ω_b) undoes it.
                let error = -wa.cross(wb);
                let inv_i = a.inv_inertia + b.inv_inertia;
                let [d0, d1] = dirs;
                let k = Mat2::from_cols(
                    Vec2::new(d0.dot(inv_i * d0), d1.dot(inv_i * d0)),
                    Vec2::new(d0.dot(inv_i * d1), d1.dot(inv_i * d1)),
                );
                if k.determinant().abs() > 1e-12 {
                    rows.align = Some(AlignRow {
                        dirs,
                        inv_k: k.inverse(),
                        bias: correction * Vec2::new(error.dot(d0), error.dot(d1)),
                    });
                }
                if let Some((min, max)) = limits {
                    let (ra, rb) = (qa * ref_a, qb * ref_b);
                    // Angle grows at (ω_b − ω_a)·wa.
                    let angle = ra.cross(rb).dot(wa).atan2(ra.dot(rb));
                    if angle - min < LIMIT_MARGIN {
                        rows.lower = limit_row(&a, &b, -wa, angle - min, dt);
                    }
                    if max - angle < LIMIT_MARGIN {
                        rows.upper = limit_row(&a, &b, wa, max - angle, dt);
                    }
                }
            }
            Constraint::Distance { length } => {
                let delta = pa - pb;
                if let Some(normal) = delta.try_normalize() {
                    let inv = a.inverse_mass_along(normal) + b.inverse_mass_along(normal);
                    if inv > 0.0 {
                        rows.length = Some(LengthRow {
                            normal,
                            mass: 1.0 / inv,
                            bias: correction * (delta.length() - length),
                        });
                    }
                }
            }
        }
        // Rows that are off this step carry no impulse into the next.
        let i = &mut rows.impulse;
        if rows.point.is_none() {
            i.point = Vec3::ZERO;
        }
        if rows.align.is_none() {
            i.angular = Vec3::ZERO;
        }
        if rows.lower.is_none() {
            i.lower = 0.0;
        }
        if rows.upper.is_none() {
            i.upper = 0.0;
        }
        if rows.length.is_none() {
            i.distance = 0.0;
        }
        out.push(rows);
    }
    out
}

impl Rows {
    fn push(&self, world: &mut World, impulse: Vec3) {
        self.a.apply(world, impulse);
        self.b.apply(world, -impulse);
    }

    fn twist(&self, world: &mut World, impulse: Vec3) {
        self.a.apply_angular(world, impulse);
        self.b.apply_angular(world, -impulse);
    }

    fn relative_spin(&self, world: &World) -> Vec3 {
        self.a.angular_velocity(world) - self.b.angular_velocity(world)
    }

    /// Re-applies last step's impulses (the alignment one re-projected onto
    /// this step's perpendicular directions).
    pub(crate) fn warm_start(&mut self, world: &mut World) {
        let i = self.impulse;
        let mut spin = i.angular;
        if let Some(align) = &self.align {
            spin = align.dirs.iter().map(|&d| d * spin.dot(d)).sum();
            self.impulse.angular = spin;
        }
        let rows = [&self.lower, &self.upper];
        for (row, lambda) in rows.into_iter().zip([i.lower, i.upper]) {
            if let Some(row) = row {
                spin += row.dir * lambda;
            }
        }
        if spin != Vec3::ZERO {
            self.twist(world, spin);
        }
        let mut push = i.point;
        if let Some(row) = &self.length {
            push += row.normal * i.distance;
        }
        if push != Vec3::ZERO {
            self.push(world, push);
        }
    }

    /// One Gauss–Seidel update: limits, then axis alignment, then the anchors
    /// (last solved wins, and a separated anchor is the most visible error).
    pub(crate) fn solve(&mut self, world: &mut World) {
        for upper in [false, true] {
            let (row, total) = if upper {
                (&self.upper, &mut self.impulse.upper)
            } else {
                (&self.lower, &mut self.impulse.lower)
            };
            if let Some(row) = row {
                let speed =
                    (self.a.angular_velocity(world) - self.b.angular_velocity(world)).dot(row.dir);
                let next = (*total + (row.bias - speed) * row.mass).max(0.0);
                let applied = next - *total;
                *total = next;
                self.a.apply_angular(world, row.dir * applied);
                self.b.apply_angular(world, -row.dir * applied);
            }
        }
        if let Some(align) = &self.align {
            let spin = self.relative_spin(world);
            let speed = Vec2::new(spin.dot(align.dirs[0]), spin.dot(align.dirs[1]));
            let total = self.impulse.angular;
            let total = Vec2::new(total.dot(align.dirs[0]), total.dot(align.dirs[1]));
            let lambda = -self.soft.mass_scale * (align.inv_k * (speed + align.bias))
                - self.soft.impulse_scale * total;
            let impulse = align.dirs[0] * lambda.x + align.dirs[1] * lambda.y;
            self.impulse.angular += impulse;
            self.twist(world, impulse);
        }
        if let Some(row) = &self.length {
            let speed = (self.a.velocity(world) - self.b.velocity(world)).dot(row.normal);
            let lambda = -self.soft.mass_scale * row.mass * (speed + row.bias)
                - self.soft.impulse_scale * self.impulse.distance;
            self.impulse.distance += lambda;
            self.push(world, row.normal * lambda);
        }
        if let Some(point) = &self.point {
            let velocity = self.a.velocity(world) - self.b.velocity(world);
            let lambda = -self.soft.mass_scale * (point.inv_k * (velocity + point.bias))
                - self.soft.impulse_scale * self.impulse.point;
            self.impulse.point += lambda;
            self.push(world, lambda);
        }
    }

    /// Saves this step's impulses on the joint for warm starting.
    pub(crate) fn store(&self, joints: &mut [Option<Joint>]) {
        if let Some(Some(joint)) = joints.get_mut(self.index) {
            joint.impulse = self.impulse;
        }
    }
}
