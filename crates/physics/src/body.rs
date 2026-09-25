use std::fmt;

use glam::{Quat, Vec3};

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
}

impl BodyDesc {
    pub fn dynamic(mass: f32, position: Vec3) -> Self {
        Self {
            kind: BodyKind::Dynamic { mass },
            position,
            orientation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
        }
    }

    pub fn fixed(position: Vec3) -> Self {
        Self {
            kind: BodyKind::Static,
            ..Self::dynamic(1.0, position)
        }
    }
}

/// Why a body could not be created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyError {
    NonFinite,
    InvalidMass,
    InvalidOrientation,
}

impl fmt::Display for BodyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(f, "position, velocity, and orientation must be finite"),
            Self::InvalidMass => write!(f, "dynamic bodies need a finite, positive mass"),
            Self::InvalidOrientation => write!(f, "orientation must be a non-zero quaternion"),
        }
    }
}

impl std::error::Error for BodyError {}

/// A rigid body's state. Fields are private so every body stays finite and valid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidBody {
    kind: BodyKind,
    position: Vec3,
    orientation: Quat,
    velocity: Vec3,
    inverse_mass: f32,
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
        })
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

    /// Semi-implicit Euler: update velocity first, then move with the new velocity.
    pub(crate) fn integrate(&mut self, gravity: Vec3, dt: f32) {
        if self.inverse_mass == 0.0 {
            return;
        }
        self.velocity += gravity * dt;
        self.position += self.velocity * dt;
        debug_assert!(
            self.position.is_finite() && self.velocity.is_finite(),
            "non-finite body state after integrate"
        );
    }
}
