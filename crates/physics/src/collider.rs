use glam::Vec3;

/// Collision shape, positioned by its body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Collider {
    Sphere {
        radius: f32,
    },
    /// Oriented box, centered on the body, axes along the body's local axes.
    Box {
        half_extents: Vec3,
    },
    /// Segment of length `2·half_height` along the body's local Y, inflated by `radius`.
    Capsule {
        half_height: f32,
        radius: f32,
    },
    /// Infinite plane through the body's position. Static bodies only; the
    /// normal points out of the solid side and is normalized on creation.
    Plane {
        normal: Vec3,
    },
}

/// Surface response. Combined per contact: restitution `max`, friction `√(a·b)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// 0 = no bounce, 1 = perfectly elastic.
    pub restitution: f32,
    /// Coulomb coefficient.
    pub friction: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            restitution: 0.3,
            friction: 0.5,
        }
    }
}

impl Material {
    pub(crate) fn is_valid(self) -> bool {
        [self.restitution, self.friction]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
    }

    /// `(restitution, friction)` for a contact between two materials.
    pub(crate) fn combine(a: Material, b: Material) -> (f32, f32) {
        (
            a.restitution.max(b.restitution),
            (a.friction * b.friction).sqrt(),
        )
    }
}
