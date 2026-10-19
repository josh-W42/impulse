//! Shape queries: sweep a shape through the world (shape casts) or list what a
//! shape overlaps. These are what character controllers, melee swings, area
//! effects, and teleports are built from.
//!
//! Shapes are handled as a core plus a radius: a sphere is a point inflated by
//! its radius, a capsule a segment; boxes and hulls have radius zero. GJK
//! distance between cores, minus the radii, gives the gap between the shapes.
//! A cast advances conservatively (Mirtich): the separating plane between the
//! shapes is `gap` away, so the shape can travel `gap / closing speed` along
//! the cast direction without touching; repeat until the gap closes.

use glam::{Quat, Vec3};

use crate::broadphase::Aabb;
use crate::collider::Collider;
use crate::distance::{Translated, closest_points};
use crate::filter::QueryFilter;
use crate::gjk::{Convex, penetration};
use crate::hull::ConvexHull;
use crate::world::{BodyHandle, World};

/// Where a shape cast first touched a body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeHit {
    pub body: BodyHandle,
    /// How far the shape travelled before touching (within [`CAST_TOLERANCE`]);
    /// zero if it started overlapping the body.
    pub distance: f32,
    /// Contact point, on the hit body's surface.
    pub point: Vec3,
    /// Unit surface normal of the hit body at `point`, pointing toward the cast
    /// shape. For a shape that started overlapping, the direction that separates
    /// it fastest.
    pub normal: Vec3,
}

/// A cast stops this close to the surface it hits (metres).
pub const CAST_TOLERANCE: f32 = 1e-4;
/// Conservative advancement converges in a handful of iterations; this bounds
/// the grazing cases.
const MAX_CAST_ITERATIONS: usize = 32;

/// A collider posed in the world as a core and a radius. Planes have no core.
enum Core {
    Shape { core: Convex<'static>, radius: f32 },
    Hull { points: Vec<Vec3> },
}

impl Core {
    fn new(
        collider: Collider,
        position: Vec3,
        orientation: Quat,
        hulls: &[ConvexHull],
    ) -> Option<Self> {
        Some(match collider {
            Collider::Sphere { radius } => Core::Shape {
                core: Convex::Sphere {
                    center: position,
                    radius: 0.0,
                },
                radius,
            },
            Collider::Capsule {
                half_height,
                radius,
            } => {
                let half = orientation * Vec3::Y * half_height;
                Core::Shape {
                    core: Convex::Capsule {
                        a: position - half,
                        b: position + half,
                        radius: 0.0,
                    },
                    radius,
                }
            }
            Collider::Box { half_extents } => Core::Shape {
                core: Convex::cuboid(position, orientation, half_extents),
                radius: 0.0,
            },
            Collider::Hull { hull } => Core::Hull {
                points: hulls
                    .get(hull.0 as usize)?
                    .vertices()
                    .iter()
                    .map(|&v| position + orientation * v)
                    .collect(),
            },
            Collider::Plane { .. } => return None,
        })
    }

    fn core(&self) -> (Convex<'_>, f32) {
        match self {
            Core::Shape { core, radius } => (*core, *radius),
            Core::Hull { points } => (Convex::Points { points }, 0.0),
        }
    }

    /// The whole shape, radius included (for penetration when cores overlap).
    fn full(&self) -> Convex<'_> {
        match self.core() {
            (Convex::Sphere { center, .. }, radius) => Convex::Sphere { center, radius },
            (Convex::Capsule { a, b, .. }, radius) => Convex::Capsule { a, b, radius },
            (core, _) => core,
        }
    }
}

/// `(distance, point, normal)` of a hit.
type Hit = (f32, Vec3, Vec3);

/// Sweeps `caster` along unit `dir` against `target`.
fn cast_core(caster: &Core, dir: Vec3, max_distance: f32, target: &Core) -> Option<Hit> {
    let (a, ra) = caster.core();
    let (b, rb) = target.core();
    let radius = ra + rb;
    let mut t = 0.0;
    for _ in 0..MAX_CAST_ITERATIONS {
        let Some((pa, pb)) = closest_points(&Translated(&a, dir * t), &b) else {
            // Cores overlap. Advancement stops short of that, so only a cast
            // that starts this deep in gets here.
            return (t == 0.0).then(|| started_inside(caster, target, dir));
        };
        let delta = pa - pb;
        let gap = delta.length() - radius;
        let normal = delta.normalize_or(-dir);
        if gap <= CAST_TOLERANCE {
            let point = pb + normal * rb;
            return Some((t, point, normal));
        }
        // How fast the shape closes on the separating plane.
        let closing = -dir.dot(normal);
        if closing <= 1e-6 {
            return None; // moving parallel to it or away
        }
        // Aim to stop short by half the tolerance: aiming at contact itself
        // lands on it exactly for flat features, and rounding then puts the
        // shapes a hair inside each other.
        t += (gap - 0.5 * CAST_TOLERANCE) / closing;
        if t > max_distance {
            return None;
        }
    }
    // Still grazing after every iteration: report the hit rather than tunnel.
    let (pa, pb) = closest_points(&Translated(&a, dir * t), &b)?;
    let normal = (pa - pb).normalize_or(-dir);
    Some((t, pb + normal * rb, normal))
}

/// A cast that starts with the cores overlapping: distance zero, separating
/// along the penetration normal.
fn started_inside(caster: &Core, target: &Core, dir: Vec3) -> Hit {
    let (a, b) = (caster.full(), target.full());
    match penetration(&a, &b) {
        Some((normal, _)) => (0.0, b.support(normal), normal),
        None => (0.0, b.support(-dir), -dir),
    }
}

/// Sweeps `caster` along unit `dir` against the plane through `origin` with
/// unit `normal` (solid below).
fn cast_plane(
    caster: &Core,
    dir: Vec3,
    max_distance: f32,
    origin: Vec3,
    normal: Vec3,
) -> Option<Hit> {
    let (core, radius) = caster.core();
    let lowest = core.support(-normal);
    let height = (lowest - origin).dot(normal);
    let gap = height - radius;
    if gap <= 0.0 {
        return Some((0.0, lowest - normal * height, normal));
    }
    let closing = -dir.dot(normal);
    if closing <= 1e-6 {
        return None;
    }
    let t = (gap - CAST_TOLERANCE).max(0.0) / closing;
    (t <= max_distance).then(|| {
        (
            t,
            lowest + dir * t - normal * (radius + CAST_TOLERANCE),
            normal,
        )
    })
}

/// Whether two posed shapes overlap (touching doesn't count).
fn overlaps(a: &Core, b: &Core) -> bool {
    let ((ca, ra), (cb, rb)) = (a.core(), b.core());
    match closest_points(&ca, &cb) {
        Some((pa, pb)) => pa.distance(pb) < ra + rb,
        None => true,
    }
}

fn valid_orientation(q: Quat) -> Option<Quat> {
    (q.is_finite() && q.length_squared() > 1e-12).then(|| q.normalize())
}

impl World {
    /// The first body a shape touches when swept from `position` along
    /// `direction` (needn't be unit length) for up to `max_distance`, ignoring
    /// bodies `filter` rejects. The shape keeps its orientation; planes can't be
    /// cast. A body the shape already overlaps is hit at distance zero.
    pub fn shape_cast(
        &self,
        shape: Collider,
        position: Vec3,
        orientation: Quat,
        direction: Vec3,
        max_distance: f32,
        filter: QueryFilter,
    ) -> Option<ShapeHit> {
        let dir = direction.try_normalize()?;
        if !(position.is_finite() && orientation.is_finite() && max_distance >= 0.0) {
            return None;
        }
        let orientation = valid_orientation(orientation)?;
        let caster = Core::new(shape, position, orientation, self.hulls())?;
        let start = Aabb::of(shape, position, orientation, self.hulls())?;
        let end = Aabb::of(
            shape,
            position + dir * max_distance,
            orientation,
            self.hulls(),
        )?;
        let mut candidates = Vec::new();
        self.tree().query(start.union(end), |h| candidates.push(h));
        candidates.extend(self.planes());
        candidates
            .into_iter()
            .filter_map(|handle| {
                let body = self
                    .body(handle)
                    .filter(|b| filter.accepts(handle, b.filter()))?;
                let (distance, point, normal) = match body.collider()? {
                    Collider::Plane { normal } => {
                        cast_plane(&caster, dir, max_distance, body.position(), normal)?
                    }
                    collider => {
                        let target =
                            Core::new(collider, body.position(), body.orientation(), self.hulls())?;
                        cast_core(&caster, dir, max_distance, &target)?
                    }
                };
                Some(ShapeHit {
                    body: handle,
                    distance,
                    point,
                    normal,
                })
            })
            .min_by(|a, b| a.distance.total_cmp(&b.distance).then(a.body.cmp(&b.body)))
    }

    /// Every body (that `filter` accepts) overlapping `shape` at the given pose,
    /// in handle order. Touching doesn't count. Planes can't be the query shape.
    pub fn overlap(
        &self,
        shape: Collider,
        position: Vec3,
        orientation: Quat,
        filter: QueryFilter,
    ) -> Vec<BodyHandle> {
        let Some(orientation) = valid_orientation(orientation).filter(|_| position.is_finite())
        else {
            return Vec::new();
        };
        let (Some(query), Some(bounds)) = (
            Core::new(shape, position, orientation, self.hulls()),
            Aabb::of(shape, position, orientation, self.hulls()),
        ) else {
            return Vec::new();
        };
        let mut candidates = Vec::new();
        self.tree().query(bounds, |h| candidates.push(h));
        candidates.extend(self.planes());
        let mut hits: Vec<BodyHandle> = candidates
            .into_iter()
            .filter(|&handle| {
                let Some(body) = self
                    .body(handle)
                    .filter(|b| filter.accepts(handle, b.filter()))
                else {
                    return false;
                };
                match body.collider() {
                    Some(Collider::Plane { normal }) => {
                        let (core, radius) = query.core();
                        (core.support(-normal) - body.position()).dot(normal) < radius
                    }
                    Some(collider) => {
                        Core::new(collider, body.position(), body.orientation(), self.hulls())
                            .is_some_and(|target| overlaps(&query, &target))
                    }
                    None => false,
                }
            })
            .collect();
        hits.sort_unstable();
        hits
    }
}
