//! Ray queries: exact ray–shape intersection, accelerated by the broadphase tree.

use glam::{Quat, Vec3};

use crate::broadphase::Aabb;
use crate::collider::Collider;
use crate::world::BodyHandle;

/// The nearest surface a ray reached.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    pub body: BodyHandle,
    /// Distance from the ray origin (the direction is normalized).
    pub t: f32,
    pub point: Vec3,
    /// Outward surface normal at `point`.
    pub normal: Vec3,
}

/// Slab test: the ray's entry distance into `aabb`, if within `max_t`.
pub(crate) fn ray_aabb(origin: Vec3, inv_dir: Vec3, aabb: Aabb, max_t: f32) -> bool {
    let t1 = (aabb.min - origin) * inv_dir;
    let t2 = (aabb.max - origin) * inv_dir;
    let t_enter = t1.min(t2).max_element().max(0.0);
    let t_exit = t1.max(t2).min_element().min(max_t);
    t_enter <= t_exit
}

/// Exact hit `(t, normal)` of a ray (unit `dir`) against a posed collider.
/// Rays starting inside a shape don't hit it.
pub(crate) fn ray_shape(
    origin: Vec3,
    dir: Vec3,
    collider: Collider,
    position: Vec3,
    orientation: Quat,
) -> Option<(f32, Vec3)> {
    match collider {
        Collider::Sphere { radius } => ray_sphere(origin, dir, position, radius),
        Collider::Box { half_extents } => {
            let inv = orientation.inverse();
            let (t, normal) = ray_local_box(inv * (origin - position), inv * dir, half_extents)?;
            Some((t, orientation * normal))
        }
        Collider::Plane { normal } => {
            let facing = dir.dot(normal);
            if facing >= 0.0 {
                return None; // parallel, or hitting the back face
            }
            let t = (position - origin).dot(normal) / facing;
            (t >= 0.0).then_some((t, normal))
        }
        Collider::Capsule {
            half_height,
            radius,
        } => {
            let axis = orientation * Vec3::Y;
            let ends = [position - axis * half_height, position + axis * half_height];
            [
                ray_cylinder(origin, dir, position, axis, half_height, radius),
                ray_sphere(origin, dir, ends[0], radius),
                ray_sphere(origin, dir, ends[1], radius),
            ]
            .into_iter()
            .flatten()
            .min_by(|a, b| a.0.total_cmp(&b.0))
        }
    }
}

fn ray_sphere(origin: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<(f32, Vec3)> {
    let m = origin - center;
    let c = m.length_squared() - radius * radius;
    if c < 0.0 {
        return None; // starts inside
    }
    let b = m.dot(dir);
    let disc = b * b - c;
    if b > 0.0 || disc < 0.0 {
        return None; // pointing away, or missing
    }
    let t = -b - disc.sqrt();
    Some((t, (m + dir * t) / radius))
}

fn ray_local_box(origin: Vec3, dir: Vec3, half: Vec3) -> Option<(f32, Vec3)> {
    let (mut t_enter, mut t_exit, mut normal) = (0.0f32, f32::INFINITY, Vec3::ZERO);
    for axis in 0..3 {
        if dir[axis].abs() < 1e-9 {
            if origin[axis].abs() > half[axis] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / dir[axis];
        let (near, far) = (
            (-half[axis] - origin[axis]) * inv,
            (half[axis] - origin[axis]) * inv,
        );
        let (t0, t1, sign) = if near < far {
            (near, far, -1.0)
        } else {
            (far, near, 1.0)
        };
        if t0 > t_enter {
            t_enter = t0;
            normal = Vec3::ZERO;
            normal[axis] = sign;
        }
        t_exit = t_exit.min(t1);
        if t_enter > t_exit {
            return None;
        }
    }
    // Zero normal: the ray started inside.
    (normal != Vec3::ZERO).then_some((t_enter, normal))
}

fn ray_cylinder(
    origin: Vec3,
    dir: Vec3,
    center: Vec3,
    axis: Vec3,
    half: f32,
    radius: f32,
) -> Option<(f32, Vec3)> {
    let m = origin - center;
    let (m_perp, d_perp) = (m - axis * m.dot(axis), dir - axis * dir.dot(axis));
    let a = d_perp.length_squared();
    if a < 1e-12 {
        return None; // along the axis: only the end spheres can be hit
    }
    let (b, c) = (
        m_perp.dot(d_perp),
        m_perp.length_squared() - radius * radius,
    );
    let disc = b * b - a * c;
    if c < 0.0 || disc < 0.0 {
        return None;
    }
    let t = (-b - disc.sqrt()) / a;
    let along = (m + dir * t).dot(axis);
    (t >= 0.0 && along.abs() <= half).then(|| (t, (m_perp + d_perp * t) / radius))
}
