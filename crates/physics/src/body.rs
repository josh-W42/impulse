use std::fmt;

use glam::{Mat3, Quat, Vec3};

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
    /// Radians per second, world space.
    pub angular_velocity: Vec3,
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
            angular_velocity: Vec3::ZERO,
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

    pub fn with_angular_velocity(self, angular_velocity: Vec3) -> Self {
        Self {
            angular_velocity,
            ..self
        }
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
                "spheres/boxes need finite positive sizes; planes a non-zero normal and a static body"
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
    pub(crate) angular_velocity: Vec3,
    inverse_mass: f32,
    /// Diagonal of the inverse inertia tensor in body space (principal axes).
    inverse_inertia_local: Vec3,
    /// Sleeping bodies skip gravity, solving, and integration until woken.
    pub(crate) asleep: bool,
    /// Seconds spent below the sleep speed thresholds.
    pub(crate) sleep_timer: f32,
    collider: Option<Collider>,
    material: Material,
}

impl RigidBody {
    pub(crate) fn new(desc: BodyDesc) -> Result<Self, BodyError> {
        if !(desc.position.is_finite()
            && desc.velocity.is_finite()
            && desc.angular_velocity.is_finite()
            && desc.orientation.is_finite())
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
        let inverse_inertia_local = inverse_inertia(collider, inverse_mass);
        let (velocity, angular_velocity) = if inverse_mass == 0.0 {
            (Vec3::ZERO, Vec3::ZERO)
        } else if inverse_inertia_local == Vec3::ZERO {
            (desc.velocity, Vec3::ZERO)
        } else {
            (desc.velocity, desc.angular_velocity)
        };
        Ok(Self {
            kind: desc.kind,
            position: desc.position,
            orientation: desc.orientation.normalize(),
            velocity,
            angular_velocity,
            inverse_mass,
            inverse_inertia_local,
            asleep: false,
            sleep_timer: 0.0,
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

    pub fn is_asleep(&self) -> bool {
        self.asleep
    }

    /// Dynamic and awake: gets integrated and solved this step.
    pub(crate) fn is_active(&self) -> bool {
        self.inverse_mass != 0.0 && !self.asleep
    }

    pub fn angular_velocity(&self) -> Vec3 {
        self.angular_velocity
    }

    /// Inverse inertia tensor in world space: `R·I⁻¹·Rᵀ`. Zero if the body can't rotate.
    pub fn inverse_inertia_world(&self) -> Mat3 {
        let r = Mat3::from_quat(self.orientation);
        r * Mat3::from_diagonal(self.inverse_inertia_local) * r.transpose()
    }

    /// First half of semi-implicit Euler: forces change velocity.
    pub(crate) fn apply_gravity(&mut self, gravity: Vec3, dt: f32) {
        if self.is_active() {
            self.velocity += gravity * dt;
        }
    }

    /// Second half: move with the (contact-corrected) new velocity.
    pub(crate) fn advance(&mut self, dt: f32) {
        if self.is_active() {
            self.position += self.velocity * dt;
            // dq/dt = ½·(ω, 0)·q; renormalize to stop drift. (No gyroscopic term.)
            let w = self.angular_velocity;
            let spin = Quat::from_xyzw(w.x, w.y, w.z, 0.0) * self.orientation * (0.5 * dt);
            self.orientation = (self.orientation + spin).normalize();
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
        Collider::Box { half_extents }
            if half_extents.is_finite() && half_extents.min_element() > 0.0 =>
        {
            Ok(collider)
        }
        Collider::Capsule {
            half_height,
            radius,
        } if half_height.is_finite()
            && half_height >= 0.0
            && radius.is_finite()
            && radius > 0.0 =>
        {
            Ok(collider)
        }
        Collider::Plane { normal } if is_static && normal.is_finite() && normal.length() > 1e-6 => {
            Ok(Collider::Plane {
                normal: normal.normalize(),
            })
        }
        _ => Err(BodyError::InvalidCollider),
    }
}

/// Inverse principal moments; zero (no rotation) for static or collider-less bodies.
fn inverse_inertia(collider: Option<Collider>, inverse_mass: f32) -> Vec3 {
    if inverse_mass == 0.0 {
        return Vec3::ZERO;
    }
    let mass = 1.0 / inverse_mass;
    let inertia = match collider {
        Some(Collider::Sphere { radius }) => Vec3::splat(0.4 * mass * radius * radius),
        Some(Collider::Box { half_extents: h }) => {
            let s = h * h;
            Vec3::new(s.y + s.z, s.x + s.z, s.x + s.y) * (mass / 3.0)
        }
        Some(Collider::Capsule {
            half_height: h,
            radius: r,
        }) => capsule_inertia(mass, h, r),
        _ => return Vec3::ZERO,
    };
    inertia.recip()
}

/// Capsule along Y: a cylinder (length 2h) plus two hemispheres, mass split by volume.
fn capsule_inertia(mass: f32, h: f32, r: f32) -> Vec3 {
    let cylinder_volume = std::f32::consts::PI * r * r * 2.0 * h;
    let sphere_volume = 4.0 / 3.0 * std::f32::consts::PI * r * r * r;
    let mc = mass * cylinder_volume / (cylinder_volume + sphere_volume);
    let ms = mass - mc;
    let axial = mc * r * r / 2.0 + ms * 0.4 * r * r;
    // Hemispheres about the capsule's center (parallel-axis to their centroids).
    let transverse = mc * (h * h / 3.0 + r * r / 4.0) + ms * (0.4 * r * r + h * h + 0.375 * h * r);
    Vec3::new(transverse, axial, transverse)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capsule_inertia_approaches_sphere_and_rod_limits() {
        // No cylinder: a solid sphere, 2/5·m·r² about every axis.
        let sphere = capsule_inertia(2.0, 0.0, 0.5);
        assert!(sphere.abs_diff_eq(Vec3::splat(0.4 * 2.0 * 0.25), 1e-6));
        // Vanishing radius: a thin rod of length 2h, m·L²/12 transverse.
        let rod = capsule_inertia(3.0, 1.0, 1e-4);
        assert!((rod.x - 3.0 * 4.0 / 12.0).abs() < 1e-3, "{rod}");
        assert!(rod.y < 1e-6);
    }

    #[test]
    fn inertia_matches_textbook_formulas() {
        // Solid sphere: I = 2/5·m·r² = 0.4·2·0.25 = 0.2.
        let sphere = inverse_inertia(Some(Collider::Sphere { radius: 0.5 }), 1.0 / 2.0);
        assert!(sphere.abs_diff_eq(Vec3::splat(5.0), 1e-5));
        // Box, m = 3, half-extents (1, 2, 3): Iₓ = m/3·(4 + 9) = 13, I_y = 10, I_z = 5.
        let cuboid = inverse_inertia(
            Some(Collider::Box {
                half_extents: Vec3::new(1.0, 2.0, 3.0),
            }),
            1.0 / 3.0,
        );
        assert!(cuboid.abs_diff_eq(Vec3::new(1.0 / 13.0, 1.0 / 10.0, 1.0 / 5.0), 1e-6));
    }

    #[test]
    fn static_and_collider_less_bodies_do_not_rotate() {
        let spin = Vec3::new(1.0, 2.0, 3.0);
        let fixed = RigidBody::new(
            BodyDesc::fixed(Vec3::ZERO)
                .with_collider(Collider::Box {
                    half_extents: Vec3::ONE,
                })
                .with_angular_velocity(spin),
        )
        .unwrap();
        let bare =
            RigidBody::new(BodyDesc::dynamic(1.0, Vec3::ZERO).with_angular_velocity(spin)).unwrap();
        for body in [fixed, bare] {
            assert_eq!(body.angular_velocity(), Vec3::ZERO);
            assert_eq!(body.inverse_inertia_world(), Mat3::ZERO);
        }
    }

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
