//! Narrowphase: exact sphere–sphere and sphere–plane tests. Brute-force pairs
//! for now; a broadphase arrives in M5.

use glam::Vec3;

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

/// `(normal b→a, depth, point)` if the spheres overlap. Touching isn't overlap.
pub(crate) fn sphere_sphere(
    center_a: Vec3,
    radius_a: f32,
    center_b: Vec3,
    radius_b: f32,
) -> Option<(Vec3, f32, Vec3)> {
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
) -> Option<(Vec3, f32, Vec3)> {
    let height = (center - plane_point).dot(plane_normal);
    let depth = radius - height;
    (depth > 0.0).then(|| (plane_normal, depth, center - plane_normal * height))
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
            let contact = match (ca, cb) {
                (Collider::Sphere { radius: ra }, Collider::Sphere { radius: rb }) => {
                    sphere_sphere(pa, ra, pb, rb).map(|g| (ha, hb, g))
                }
                (Collider::Sphere { radius }, Collider::Plane { normal }) => {
                    sphere_plane(pa, radius, pb, normal).map(|g| (ha, hb, g))
                }
                (Collider::Plane { normal }, Collider::Sphere { radius }) => {
                    sphere_plane(pb, radius, pa, normal).map(|g| (hb, ha, g))
                }
                (Collider::Plane { .. }, Collider::Plane { .. }) => None,
            };
            if let Some((a, b, (normal, depth, point))) = contact {
                contacts.push(Contact {
                    a,
                    b,
                    normal,
                    depth,
                    point,
                });
            }
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
