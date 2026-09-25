//! Narrowphase: exact sphere–sphere and sphere–plane tests. Brute-force pairs
//! for now; a broadphase arrives in M5.

use glam::{Quat, Vec3};

use crate::box_box::box_box;
use crate::collider::Collider;
use crate::world::{BodyHandle, World};

/// Overlap between two bodies. `normal` points from `b` toward `a`, so moving
/// `a` along it (or `b` against it) separates them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Contact {
    pub a: BodyHandle,
    pub b: BodyHandle,
    pub normal: Vec3,
    pub depth: f32,
    pub point: Vec3,
}

/// `(normal b→a, depth, point)` for one contact point.
pub(crate) type Geometry = (Vec3, f32, Vec3);

/// `(normal b→a, depth, point)` if the spheres overlap. Touching isn't overlap.
pub(crate) fn sphere_sphere(
    center_a: Vec3,
    radius_a: f32,
    center_b: Vec3,
    radius_b: f32,
) -> Option<Geometry> {
    let offset = center_a - center_b;
    let reach = radius_a + radius_b;
    let distance_sq = offset.length_squared();
    if distance_sq >= reach * reach {
        return None;
    }
    let distance = distance_sq.sqrt();
    // Coincident centers have no meaningful direction; any unit normal separates them.
    let normal = if distance > 1e-6 {
        offset / distance
    } else {
        Vec3::Y
    };
    let depth = reach - distance;
    let point = center_b + normal * (radius_b - depth * 0.5);
    Some((normal, depth, point))
}

/// Sphere against a plane through `plane_point` with unit `plane_normal`.
pub(crate) fn sphere_plane(
    center: Vec3,
    radius: f32,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Option<Geometry> {
    let height = (center - plane_point).dot(plane_normal);
    let depth = radius - height;
    (depth > 0.0).then(|| (plane_normal, depth, center - plane_normal * height))
}

/// Contacts where box corners are below the plane (up to 4 for a resting box).
pub(crate) fn box_plane(
    center: Vec3,
    orientation: Quat,
    half_extents: Vec3,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Vec<Geometry> {
    let mut hits = Vec::new();
    for i in 0..8 {
        let sign = |bit: usize| if i & bit == 0 { -1.0 } else { 1.0 };
        let local = half_extents * Vec3::new(sign(1), sign(2), sign(4));
        let corner = center + orientation * local;
        let height = (corner - plane_point).dot(plane_normal);
        if height < 0.0 {
            let depth = -height;
            hits.push((plane_normal, depth, corner + plane_normal * (depth * 0.5)));
        }
    }
    hits
}

/// Sphere against an oriented box. The normal points from the box to the sphere.
pub(crate) fn sphere_box(
    sphere_center: Vec3,
    radius: f32,
    box_center: Vec3,
    orientation: Quat,
    half_extents: Vec3,
) -> Option<Geometry> {
    let local = orientation.inverse() * (sphere_center - box_center);
    let closest = local.clamp(-half_extents, half_extents);
    let (normal_local, depth, point_local) = if closest == local {
        // Center inside the box: leave through the nearest face.
        let gaps = half_extents - local.abs();
        let axis = if gaps.x <= gaps.y && gaps.x <= gaps.z {
            0
        } else if gaps.y <= gaps.z {
            1
        } else {
            2
        };
        let sign = if local[axis] < 0.0 { -1.0 } else { 1.0 };
        let mut normal = Vec3::ZERO;
        normal[axis] = sign;
        let mut surface = local;
        surface[axis] = sign * half_extents[axis];
        (normal, radius + gaps[axis], surface)
    } else {
        let offset = local - closest;
        let distance = offset.length();
        if distance >= radius {
            return None;
        }
        (offset / distance, radius - distance, closest)
    };
    Some((
        orientation * normal_local,
        depth,
        box_center + orientation * point_local,
    ))
}

/// Every overlapping pair in the world (O(n²)).
pub fn find_contacts(world: &World) -> Vec<Contact> {
    let bodies: Vec<_> = world
        .bodies()
        .filter_map(|(h, b)| Some((h, b, b.collider()?)))
        .collect();
    let mut contacts = Vec::new();
    for (i, &(ha, a, ca)) in bodies.iter().enumerate() {
        for &(hb, b, cb) in &bodies[i + 1..] {
            if a.inverse_mass() == 0.0 && b.inverse_mass() == 0.0 {
                continue;
            }
            let (pa, pb) = (a.position(), b.position());
            let (qa, qb) = (a.orientation(), b.orientation());
            // `flipped`: the narrowphase saw (b, a), so its normal points from a to b.
            let (hits, flipped): (Vec<Geometry>, bool) = match (ca, cb) {
                (Collider::Sphere { radius: ra }, Collider::Sphere { radius: rb }) => {
                    (sphere_sphere(pa, ra, pb, rb).into_iter().collect(), false)
                }
                (Collider::Sphere { radius }, Collider::Plane { normal }) => (
                    sphere_plane(pa, radius, pb, normal).into_iter().collect(),
                    false,
                ),
                (Collider::Plane { normal }, Collider::Sphere { radius }) => (
                    sphere_plane(pb, radius, pa, normal).into_iter().collect(),
                    true,
                ),
                (Collider::Box { half_extents }, Collider::Plane { normal }) => {
                    (box_plane(pa, qa, half_extents, pb, normal), false)
                }
                (Collider::Plane { normal }, Collider::Box { half_extents }) => {
                    (box_plane(pb, qb, half_extents, pa, normal), true)
                }
                (Collider::Sphere { radius }, Collider::Box { half_extents }) => (
                    sphere_box(pa, radius, pb, qb, half_extents)
                        .into_iter()
                        .collect(),
                    false,
                ),
                (Collider::Box { half_extents }, Collider::Sphere { radius }) => (
                    sphere_box(pb, radius, pa, qa, half_extents)
                        .into_iter()
                        .collect(),
                    true,
                ),
                (Collider::Box { half_extents: ha }, Collider::Box { half_extents: hb }) => {
                    (box_box(pa, qa, ha, pb, qb, hb), false)
                }
                (Collider::Plane { .. }, Collider::Plane { .. }) => (Vec::new(), false),
            };
            let (a, b) = if flipped { (hb, ha) } else { (ha, hb) };
            contacts.extend(hits.into_iter().map(|(normal, depth, point)| Contact {
                a,
                b,
                normal,
                depth,
                point,
            }));
        }
    }
    contacts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_spheres() {
        let (normal, depth, point) =
            sphere_sphere(Vec3::new(1.5, 0.0, 0.0), 1.0, Vec3::ZERO, 1.0).unwrap();
        assert_eq!(normal, Vec3::X);
        assert!((depth - 0.5).abs() < 1e-6);
        assert!(point.abs_diff_eq(Vec3::new(0.75, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn touching_or_apart_spheres_have_no_contact() {
        assert!(sphere_sphere(Vec3::new(2.0, 0.0, 0.0), 1.0, Vec3::ZERO, 1.0).is_none());
        assert!(sphere_sphere(Vec3::new(5.0, 0.0, 0.0), 1.0, Vec3::ZERO, 1.0).is_none());
    }

    #[test]
    fn coincident_spheres_get_a_finite_normal() {
        let (normal, depth, _) = sphere_sphere(Vec3::ONE, 0.5, Vec3::ONE, 0.5).unwrap();
        assert!(normal.is_finite() && (normal.length() - 1.0).abs() < 1e-6);
        assert!((depth - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sphere_sinking_into_plane() {
        let (normal, depth, point) =
            sphere_plane(Vec3::new(3.0, 0.25, 0.0), 0.5, Vec3::ZERO, Vec3::Y).unwrap();
        assert_eq!(normal, Vec3::Y);
        assert!((depth - 0.25).abs() < 1e-6);
        assert!(point.abs_diff_eq(Vec3::new(3.0, 0.0, 0.0), 1e-6));
        assert!(sphere_plane(Vec3::new(0.0, 0.6, 0.0), 0.5, Vec3::ZERO, Vec3::Y).is_none());
    }

    const H: Vec3 = Vec3::splat(0.5);

    #[test]
    fn flat_box_on_plane_has_four_contacts() {
        let hits = box_plane(
            Vec3::new(0.0, 0.4, 0.0),
            Quat::IDENTITY,
            H,
            Vec3::ZERO,
            Vec3::Y,
        );
        assert_eq!(hits.len(), 4);
        for (normal, depth, _) in hits {
            assert_eq!(normal, Vec3::Y);
            assert!((depth - 0.1).abs() < 1e-6);
        }
    }

    #[test]
    fn corner_down_box_touches_at_one_point() {
        let corner_down = Quat::from_rotation_arc(Vec3::NEG_ONE.normalize(), Vec3::NEG_Y);
        let hits = box_plane(
            Vec3::new(0.0, 0.8, 0.0),
            corner_down,
            H,
            Vec3::ZERO,
            Vec3::Y,
        );
        assert_eq!(hits.len(), 1);
        assert!((hits[0].1 - (3f32.sqrt() * 0.5 - 0.8)).abs() < 1e-5);
    }

    #[test]
    fn sphere_against_box_face_edge_and_corner() {
        let face =
            sphere_box(Vec3::new(0.9, 0.0, 0.0), 0.5, Vec3::ZERO, Quat::IDENTITY, H).unwrap();
        assert_eq!(face.0, Vec3::X);
        assert!((face.1 - 0.1).abs() < 1e-6);
        assert!(face.2.abs_diff_eq(Vec3::new(0.5, 0.0, 0.0), 1e-6));

        let edge =
            sphere_box(Vec3::new(0.8, 0.8, 0.0), 0.5, Vec3::ZERO, Quat::IDENTITY, H).unwrap();
        assert!(
            edge.0
                .abs_diff_eq(Vec3::new(1.0, 1.0, 0.0).normalize(), 1e-5)
        );
        assert!((edge.1 - (0.5 - 0.18f32.sqrt())).abs() < 1e-5);

        let far_corner = Vec3::splat(0.8);
        assert!(sphere_box(far_corner, 0.5, Vec3::ZERO, Quat::IDENTITY, H).is_none());
        let corner = sphere_box(far_corner, 0.6, Vec3::ZERO, Quat::IDENTITY, H).unwrap();
        assert!(corner.0.abs_diff_eq(Vec3::ONE.normalize(), 1e-5));
    }

    #[test]
    fn sphere_center_inside_box_leaves_through_nearest_face() {
        let (normal, depth, _) =
            sphere_box(Vec3::new(0.3, 0.0, 0.0), 0.5, Vec3::ZERO, Quat::IDENTITY, H).unwrap();
        assert_eq!(normal, Vec3::X);
        assert!((depth - 0.7).abs() < 1e-6);
    }

    #[test]
    fn sphere_against_rotated_box_uses_box_space() {
        // Long x-axis turned to lie along z.
        let turned = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let long = Vec3::new(1.0, 0.5, 0.5);
        let (normal, depth, _) =
            sphere_box(Vec3::new(0.0, 0.0, 1.3), 0.5, Vec3::ZERO, turned, long).unwrap();
        assert!(normal.abs_diff_eq(Vec3::Z, 1e-5));
        assert!((depth - 0.2).abs() < 1e-5);
    }

    #[test]
    fn find_contacts_orders_sphere_first_and_skips_static_pairs() {
        use crate::body::BodyDesc;
        let mut world = World::default();
        let floor = world
            .add_body(
                BodyDesc::fixed(Vec3::ZERO).with_collider(Collider::Plane { normal: Vec3::Y }),
            )
            .unwrap();
        world
            .add_body(BodyDesc::fixed(Vec3::X).with_collider(Collider::Sphere { radius: 5.0 }))
            .unwrap();
        let ball = world
            .add_body(
                BodyDesc::dynamic(1.0, Vec3::new(20.0, 0.4, 0.0))
                    .with_collider(Collider::Sphere { radius: 0.5 }),
            )
            .unwrap();
        let contacts = find_contacts(&world);
        assert_eq!(
            contacts.len(),
            1,
            "static pairs must be skipped: {contacts:?}"
        );
        assert_eq!((contacts[0].a, contacts[0].b), (ball, floor));
        assert_eq!(contacts[0].normal, Vec3::Y);
    }
}
