use std::fmt;

use glam::{Quat, Vec3};

use crate::collider::{Collider, Material};

/// How a body participates in the simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BodyKind {
    /// Moved by forces. `mass` must be finite and positive.
    Dynamic { mass: f32 },
    /// Never moves (floors, walls).
    Static,
}

/// Everything needed to create a body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyDesc {
    pub kind: BodyKind,
    pub position: Vec3,
    pub orientation: Quat,
    pub velocity: Vec3,
    /// `None`: the body moves but never collides.
    pub collider: Option<Collider>,
    pub material: Material,
}

impl BodyDesc {
    pub fn dynamic(mass: f32, position: Vec3) -> Self {
        Self {
            kind: BodyKind::Dynamic { mass },
            position,
            orientation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            collider: None,
            material: Material::default(),
        }
    }

    pub fn fixed(position: Vec3) -> Self {
        Self {
            kind: BodyKind::Static,
            ..Self::dynamic(1.0, position)
        }
    }

    pub fn with_collider(self, collider: Collider) -> Self {
        Self {
            collider: Some(collider),
            ..self
        }
    }

    pub fn with_material(self, material: Material) -> Self {
        Self { material, ..self }
    }

    pub fn with_velocity(self, velocity: Vec3) -> Self {
        Self { velocity, ..self }
    }
}

/// Why a body could not be created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyError {
    NonFinite,
    InvalidMass,
    InvalidOrientation,
    InvalidCollider,
    InvalidMaterial,
}

impl fmt::Display for BodyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(f, "position, velocity, and orientation must be finite"),
            Self::InvalidMass => write!(f, "dynamic bodies need a finite, positive mass"),
            Self::InvalidOrientation => write!(f, "orientation must be a non-zero quaternion"),
            Self::InvalidCollider => write!(
                f,
                "spheres need a finite positive radius; planes a non-zero normal and a static body"
            ),
            Self::InvalidMaterial => write!(f, "restitution and friction must be finite and >= 0"),
        }
    }
}

impl std::error::Error for BodyError {}

/// A rigid body's state. Fields are crate-private so every body stays finite and valid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidBody {
    kind: BodyKind,
    pub(crate) position: Vec3,
    orientation: Quat,
    pub(crate) velocity: Vec3,
    inverse_mass: f32,
    collider: Option<Collider>,
    material: Material,
}

impl RigidBody {
    pub(crate) fn new(desc: BodyDesc) -> Result<Self, BodyError> {
        if !(desc.position.is_finite() && desc.velocity.is_finite() && desc.orientation.is_finite())
        {
            return Err(BodyError::NonFinite);
        }
        if desc.orientation.length_squared() < 1e-12 {
            return Err(BodyError::InvalidOrientation);
        }
        let inverse_mass = match desc.kind {
            BodyKind::Dynamic { mass } if mass.is_finite() && mass > 0.0 => 1.0 / mass,
            BodyKind::Dynamic { .. } => return Err(BodyError::InvalidMass),
            BodyKind::Static => 0.0,
        };
        if !desc.material.is_valid() {
            return Err(BodyError::InvalidMaterial);
        }
        let collider = desc
            .collider
            .map(|c| validate_collider(c, inverse_mass == 0.0))
            .transpose()?;
        let velocity = if inverse_mass == 0.0 {
            Vec3::ZERO
        } else {
            desc.velocity
        };
        Ok(Self {
            kind: desc.kind,
            position: desc.position,
            orientation: desc.orientation.normalize(),
            velocity,
            inverse_mass,
            collider,
            material: desc.material,
        })
    }

    pub fn collider(&self) -> Option<Collider> {
        self.collider
    }

    pub fn material(&self) -> Material {
        self.material
    }

    pub fn kind(&self) -> BodyKind {
        self.kind
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn orientation(&self) -> Quat {
        self.orientation
    }

    pub fn velocity(&self) -> Vec3 {
        self.velocity
    }

    /// Zero for static bodies.
    pub fn inverse_mass(&self) -> f32 {
        self.inverse_mass
    }

    /// First half of semi-implicit Euler: forces change velocity.
    pub(crate) fn apply_gravity(&mut self, gravity: Vec3, dt: f32) {
        if self.inverse_mass != 0.0 {
            self.velocity += gravity * dt;
        }
    }

    /// Second half: move with the (contact-corrected) new velocity.
    pub(crate) fn advance(&mut self, dt: f32) {
        if self.inverse_mass != 0.0 {
            self.position += self.velocity * dt;
        }
        debug_assert!(
            self.position.is_finite() && self.velocity.is_finite(),
            "non-finite body state after advance"
        );
    }
}

fn validate_collider(collider: Collider, is_static: bool) -> Result<Collider, BodyError> {
    match collider {
        Collider::Sphere { radius } if radius.is_finite() && radius > 0.0 => Ok(collider),
        Collider::Plane { normal } if is_static && normal.is_finite() && normal.length() > 1e-6 => {
            Ok(Collider::Plane {
                normal: normal.normalize(),
            })
        }
        _ => Err(BodyError::InvalidCollider),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(radius: f32) -> BodyDesc {
        BodyDesc::dynamic(1.0, Vec3::ZERO).with_collider(Collider::Sphere { radius })
    }

    #[test]
    fn rejects_bad_colliders() {
        for radius in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                RigidBody::new(sphere(radius)),
                Err(BodyError::InvalidCollider)
            );
        }
        let plane = |normal| Collider::Plane { normal };
        assert_eq!(
            RigidBody::new(BodyDesc::fixed(Vec3::ZERO).with_collider(plane(Vec3::ZERO))),
            Err(BodyError::InvalidCollider)
        );
        assert_eq!(
            RigidBody::new(BodyDesc::dynamic(1.0, Vec3::ZERO).with_collider(plane(Vec3::Y))),
            Err(BodyError::InvalidCollider),
            "planes must be static"
        );
    }

    #[test]
    fn rejects_bad_materials() {
        for (restitution, friction) in [
            (-0.1, 0.5),
            (0.5, -1.0),
            (f32::NAN, 0.5),
            (0.5, f32::INFINITY),
        ] {
            let desc = sphere(1.0).with_material(Material {
                restitution,
                friction,
            });
            assert_eq!(RigidBody::new(desc), Err(BodyError::InvalidMaterial));
        }
    }

    #[test]
    fn plane_normals_are_normalized() {
        let body = RigidBody::new(BodyDesc::fixed(Vec3::ZERO).with_collider(Collider::Plane {
            normal: Vec3::new(0.0, 3.0, 4.0),
        }))
        .unwrap();
        let Some(Collider::Plane { normal }) = body.collider() else {
            panic!("expected a plane");
        };
        assert!((normal.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn materials_combine_by_max_and_geometric_mean() {
        let a = Material {
            restitution: 0.2,
            friction: 0.25,
        };
        let b = Material {
            restitution: 0.8,
            friction: 1.0,
        };
        assert_eq!(Material::combine(a, b), (0.8, 0.5));
    }
}
