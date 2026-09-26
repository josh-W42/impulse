//! Scenes as data: a serializable description (RON on disk) and the one path
//! that turns it into a physics world. Vectors are plain arrays so serde stays
//! out of the physics crate.

use std::fmt;

use glam::{EulerRot, Quat, Vec3};
use impulse_physics::{
    BodyDesc, BodyError, BodyHandle, Collider, HullError, JointDesc, JointError, Material, World,
};
use serde::{Deserialize, Serialize};

use crate::models::{ModelRegistry, model_collider};
use crate::pose::Pose;
use crate::scene::{STATIC_MATERIAL, SceneObject, pit_planes};

/// Bumped when the file format changes incompatibly.
pub const SCENE_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneDesc {
    pub version: u32,
    pub name: String,
    pub arena: Arena,
    pub objects: Vec<ObjectDesc>,
    #[serde(default)]
    pub joints: Vec<JointSpec>,
}

/// The static surroundings: a floor, optionally walled in.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Arena {
    /// Floor plus four walls at ±`half` on x and z.
    Pit { half: f32 },
    /// An infinite floor plane.
    Floor,
    /// No surroundings: only the scene's own objects (e.g. a floor object).
    Empty,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ObjectDesc {
    pub name: String,
    /// `None`: an invisible, collision-free body (e.g. a rope's hook point).
    pub shape: Option<ShapeDesc>,
    pub position: [f32; 3],
    /// Euler angles in degrees, applied X then Y then Z.
    pub rotation_deg: [f32; 3],
    pub body: BodyType,
    pub restitution: f32,
    pub friction: f32,
    pub color: [f32; 3],
    pub velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    /// Surface for built-in shapes (models use their file's material).
    pub roughness: f32,
    pub metallic: f32,
}

impl Default for ObjectDesc {
    fn default() -> Self {
        let material = Material::default();
        Self {
            name: String::new(),
            shape: None,
            position: [0.0; 3],
            rotation_deg: [0.0; 3],
            body: BodyType::Dynamic { mass: 1.0 },
            restitution: material.restitution,
            friction: material.friction,
            color: [0.7, 0.7, 0.7],
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            roughness: impulse_render::DEFAULT_ROUGHNESS,
            metallic: impulse_render::DEFAULT_METALLIC,
        }
    }
}

impl ObjectDesc {
    pub fn orientation(&self) -> Quat {
        let [x, y, z] = self.rotation_deg.map(f32::to_radians);
        Quat::from_euler(EulerRot::XYZ, x, y, z)
    }

    pub fn set_orientation(&mut self, q: Quat) {
        let (x, y, z) = q.to_euler(EulerRot::XYZ);
        // `+ 0.0` turns -0 into 0 for display.
        self.rotation_deg = [x, y, z].map(|a| a.to_degrees() + 0.0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BodyType {
    Dynamic { mass: f32 },
    Static,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShapeDesc {
    Sphere {
        radius: f32,
    },
    Box {
        half_extents: [f32; 3],
    },
    Capsule {
        half_height: f32,
        radius: f32,
    },
    /// Convex hull of these body-space points.
    Hull {
        points: Vec<[f32; 3]>,
    },
    /// A model from the registry (bundled or imported), uniformly scaled,
    /// colliding as a convex shape derived from its mesh.
    Model {
        name: String,
        scale: f32,
        #[serde(default)]
        collider: ModelCollider,
    },
}

/// How a model collides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ModelCollider {
    /// Convex hull of the mesh's extreme points.
    #[default]
    Hull,
    /// Fitted to the mesh bounds.
    Box,
    Sphere,
    /// Y-aligned.
    Capsule,
}

impl ShapeDesc {
    /// The description of a primitive collider; `None` for hulls and planes.
    pub fn primitive(collider: Collider) -> Option<Self> {
        match collider {
            Collider::Sphere { radius } => Some(Self::Sphere { radius }),
            Collider::Box { half_extents } => Some(Self::Box {
                half_extents: half_extents.to_array(),
            }),
            Collider::Capsule {
                half_height,
                radius,
            } => Some(Self::Capsule {
                half_height,
                radius,
            }),
            Collider::Hull { .. } | Collider::Plane { .. } => None,
        }
    }
}

/// A joint between objects `a` and `b` (indices into `objects`). Points and
/// axes are world space at the objects' authored poses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointSpec {
    pub a: usize,
    pub b: usize,
    pub kind: JointKindSpec,
    #[serde(default)]
    pub collide_connected: bool,
}

impl JointSpec {
    fn new(a: usize, b: usize, kind: JointKindSpec) -> Self {
        Self {
            a,
            b,
            kind,
            collide_connected: false,
        }
    }

    pub fn ball_socket(a: usize, b: usize, pivot: Vec3) -> Self {
        let kind = JointKindSpec::BallSocket {
            pivot: pivot.to_array(),
            cone: None,
        };
        Self::new(a, b, kind)
    }

    pub fn hinge(a: usize, b: usize, pivot: Vec3, axis: Vec3) -> Self {
        let kind = JointKindSpec::Hinge {
            pivot: pivot.to_array(),
            axis: axis.to_array(),
            limits_deg: None,
        };
        Self::new(a, b, kind)
    }

    pub fn distance(a: usize, b: usize, anchor_a: Vec3, anchor_b: Vec3) -> Self {
        let kind = JointKindSpec::Distance {
            anchor_a: anchor_a.to_array(),
            anchor_b: anchor_b.to_array(),
        };
        Self::new(a, b, kind)
    }

    /// Swing limit (radians). Ball-sockets only; ignored otherwise.
    pub fn with_cone(mut self, axis: Vec3, max_angle: f32) -> Self {
        if let JointKindSpec::BallSocket { cone, .. } = &mut self.kind {
            *cone = Some((axis.to_array(), max_angle.to_degrees()));
        }
        self
    }

    /// Angle limits (radians). Hinges only; ignored otherwise.
    pub fn with_limits(mut self, min: f32, max: f32) -> Self {
        if let JointKindSpec::Hinge { limits_deg, .. } = &mut self.kind {
            *limits_deg = Some((min.to_degrees(), max.to_degrees()));
        }
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JointKindSpec {
    BallSocket {
        pivot: [f32; 3],
        /// Swing limit: (axis, max angle in degrees).
        #[serde(default)]
        cone: Option<([f32; 3], f32)>,
    },
    Hinge {
        pivot: [f32; 3],
        axis: [f32; 3],
        /// (min, max) angle of `b` relative to `a`, in degrees.
        #[serde(default)]
        limits_deg: Option<(f32, f32)>,
    },
    Distance {
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
    },
}

/// Why a scene could not be loaded or built. The caller keeps its current scene.
#[derive(Debug, Clone, PartialEq)]
pub enum SceneError {
    Parse(String),
    UnsupportedVersion(u32),
    InvalidArena,
    Object {
        index: usize,
        name: String,
        error: ObjectError,
    },
    Joint {
        index: usize,
        error: JointSpecError,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObjectError {
    Body(BodyError),
    Hull(HullError),
    /// The named model isn't in the registry (not bundled, not imported).
    MissingModel(String),
    /// Model scale must be finite and positive.
    InvalidScale,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointSpecError {
    /// The joint names an object index past the end of `objects`.
    NoSuchObject(usize),
    Joint(JointError),
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(message) => write!(f, "not a valid scene file: {message}"),
            Self::UnsupportedVersion(v) => write!(
                f,
                "scene version {v} is not supported (expected {SCENE_VERSION})"
            ),
            Self::InvalidArena => write!(f, "the pit half-width must be finite and positive"),
            Self::Object { index, name, error } => {
                write!(f, "object {index} '{name}': ")?;
                match error {
                    ObjectError::Body(e) => write!(f, "{e}"),
                    ObjectError::Hull(e) => write!(f, "{e}"),
                    ObjectError::MissingModel(model) => {
                        write!(f, "model '{model}' isn't loaded; import it first")
                    }
                    ObjectError::InvalidScale => write!(f, "scale must be positive"),
                }
            }
            Self::Joint { index, error } => {
                write!(f, "joint {index}: ")?;
                match error {
                    JointSpecError::NoSuchObject(i) => write!(f, "there is no object {i}"),
                    JointSpecError::Joint(e) => write!(f, "{e}"),
                }
            }
        }
    }
}

impl std::error::Error for SceneError {}

pub fn to_ron(desc: &SceneDesc) -> String {
    ron::ser::to_string_pretty(desc, ron::ser::PrettyConfig::default())
        .expect("scene descriptions always serialize")
}

/// Largest scene file accepted (bytes); guards against loading something huge.
pub const MAX_SCENE_BYTES: usize = 16 * 1024 * 1024;

/// A safe file stem for a scene name: letters, digits, `-` and `_` (spaces
/// become `_`), at most 64 characters. `None` if nothing usable remains. Keeps
/// saved files inside the scenes folder (no separators, no `..`).
pub fn file_stem(name: &str) -> Option<String> {
    let stem: String = name
        .trim()
        .chars()
        .filter_map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => Some(c),
            ' ' => Some('_'),
            _ => None,
        })
        .take(64)
        .collect();
    (!stem.is_empty()).then_some(stem)
}

pub fn from_ron(text: &str) -> Result<SceneDesc, SceneError> {
    if text.len() > MAX_SCENE_BYTES {
        return Err(SceneError::Parse(format!(
            "file is larger than {} MB",
            MAX_SCENE_BYTES / (1024 * 1024)
        )));
    }
    let desc: SceneDesc = ron::from_str(text).map_err(|e| SceneError::Parse(e.to_string()))?;
    if desc.version != SCENE_VERSION {
        return Err(SceneError::UnsupportedVersion(desc.version));
    }
    Ok(desc)
}

fn vec3(v: [f32; 3]) -> Vec3 {
    Vec3::from_array(v)
}

/// Builds a fresh world from `desc`. `objects[i]` draws `desc.objects[i]`
/// (anchors included, though they have nothing to draw).
pub fn build_world(
    desc: &SceneDesc,
    models: &ModelRegistry,
) -> Result<(World, Vec<SceneObject>), SceneError> {
    let mut world = World::default();
    let objects = populate(&mut world, desc, models)?;
    Ok((world, objects))
}

/// Replaces `world`'s contents with `desc`, keeping its settings; every old
/// handle goes stale. On error `world` is untouched (validated off to the side).
pub fn rebuild(
    world: &mut World,
    desc: &SceneDesc,
    models: &ModelRegistry,
) -> Result<Vec<SceneObject>, SceneError> {
    build_world(desc, models)?;
    world.clear();
    Ok(populate(world, desc, models).expect("validated above"))
}

/// Adds `desc` to an empty `world`.
fn populate(
    world: &mut World,
    desc: &SceneDesc,
    models: &ModelRegistry,
) -> Result<Vec<SceneObject>, SceneError> {
    add_arena(world, desc.arena)?;
    let mut objects = Vec::with_capacity(desc.objects.len());
    for (index, object) in desc.objects.iter().enumerate() {
        let fail = |error| SceneError::Object {
            index,
            name: object.name.clone(),
            error,
        };
        let mut model = None;
        let collider = match &object.shape {
            None => None,
            Some(ShapeDesc::Sphere { radius }) => Some(Collider::Sphere { radius: *radius }),
            Some(ShapeDesc::Box { half_extents }) => Some(Collider::Box {
                half_extents: vec3(*half_extents),
            }),
            Some(ShapeDesc::Capsule {
                half_height,
                radius,
            }) => Some(Collider::Capsule {
                half_height: *half_height,
                radius: *radius,
            }),
            Some(ShapeDesc::Hull { points }) => {
                let points: Vec<Vec3> = points.iter().copied().map(vec3).collect();
                let hull = world
                    .add_hull(&points)
                    .map_err(|e| fail(ObjectError::Hull(e)))?;
                Some(Collider::Hull { hull })
            }
            Some(ShapeDesc::Model {
                name,
                scale,
                collider,
            }) => {
                let missing = || fail(ObjectError::MissingModel(name.clone()));
                let id = models.id(name).ok_or_else(missing)?;
                let asset = models.get(id).ok_or_else(missing)?;
                let (shape, visual) =
                    model_collider(id, asset, *scale, *collider, world).map_err(fail)?;
                model = Some(visual);
                Some(shape)
            }
        };
        let position = vec3(object.position);
        let orientation = object.orientation();
        let base = match object.body {
            BodyType::Dynamic { mass } => BodyDesc::dynamic(mass, position),
            BodyType::Static => BodyDesc::fixed(position),
        };
        let mut body = BodyDesc {
            orientation,
            ..base
        }
        .with_material(Material {
            restitution: object.restitution,
            friction: object.friction,
        })
        .with_velocity(vec3(object.velocity))
        .with_angular_velocity(vec3(object.angular_velocity));
        if let Some(collider) = collider {
            body = body.with_collider(collider);
        }
        let handle = world
            .add_body(body)
            .map_err(|e| fail(ObjectError::Body(e)))?;
        let [r, g, b] = object.color;
        objects.push(SceneObject {
            handle,
            color: [r, g, b, 1.0],
            prev: Pose {
                position,
                orientation,
            },
            model,
            roughness: object.roughness,
            metallic: object.metallic,
        });
    }
    for (index, joint) in desc.joints.iter().enumerate() {
        let fail = |error| SceneError::Joint { index, error };
        let handle = |i: usize| -> Result<BodyHandle, SceneError> {
            objects
                .get(i)
                .map(|o| o.handle)
                .ok_or(fail(JointSpecError::NoSuchObject(i)))
        };
        let (a, b) = (handle(joint.a)?, handle(joint.b)?);
        let spec = match &joint.kind {
            JointKindSpec::BallSocket { pivot, cone } => {
                let desc = JointDesc::ball_socket(a, b, vec3(*pivot));
                match cone {
                    Some((axis, degrees)) => desc.with_cone(vec3(*axis), degrees.to_radians()),
                    None => desc,
                }
            }
            JointKindSpec::Hinge {
                pivot,
                axis,
                limits_deg,
            } => {
                let desc = JointDesc::hinge(a, b, vec3(*pivot), vec3(*axis));
                match limits_deg {
                    Some((min, max)) => desc.with_limits(min.to_radians(), max.to_radians()),
                    None => desc,
                }
            }
            JointKindSpec::Distance { anchor_a, anchor_b } => {
                JointDesc::distance(a, b, vec3(*anchor_a), vec3(*anchor_b))
            }
        };
        world
            .add_joint(spec.with_collide_connected(joint.collide_connected))
            .map_err(|e| fail(JointSpecError::Joint(e)))?;
    }
    Ok(objects)
}

fn add_arena(world: &mut World, arena: Arena) -> Result<(), SceneError> {
    let planes = match arena {
        Arena::Pit { half } if half.is_finite() && half > 0.0 => pit_planes(half).to_vec(),
        Arena::Pit { .. } => return Err(SceneError::InvalidArena),
        Arena::Floor => vec![(Vec3::ZERO, Vec3::Y)],
        Arena::Empty => Vec::new(),
    };
    for (point, normal) in planes {
        world
            .add_body(
                BodyDesc::fixed(point)
                    .with_collider(Collider::Plane { normal })
                    .with_material(STATIC_MATERIAL),
            )
            .expect("arena planes are valid");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ModelRegistry;

    fn object(shape: Option<ShapeDesc>, position: [f32; 3], body: BodyType) -> ObjectDesc {
        ObjectDesc {
            name: "thing".into(),
            shape,
            position,
            body,
            ..ObjectDesc::default()
        }
    }

    /// One of everything: each shape, both body types, an anchor, each joint kind.
    fn sampler() -> SceneDesc {
        let dynamic = BodyType::Dynamic { mass: 2.0 };
        SceneDesc {
            version: SCENE_VERSION,
            name: "sampler".into(),
            arena: Arena::Pit { half: 6.0 },
            objects: vec![
                object(None, [0.0, 5.0, 0.0], BodyType::Static),
                object(
                    Some(ShapeDesc::Sphere { radius: 0.3 }),
                    [0.0, 4.0, 0.0],
                    dynamic,
                ),
                object(
                    Some(ShapeDesc::Box {
                        half_extents: [0.5, 0.2, 0.3],
                    }),
                    [2.0, 1.0, 0.0],
                    dynamic,
                ),
                object(
                    Some(ShapeDesc::Capsule {
                        half_height: 0.4,
                        radius: 0.2,
                    }),
                    [-2.0, 1.0, 0.0],
                    BodyType::Static,
                ),
                ObjectDesc {
                    rotation_deg: [10.0, 20.0, 30.0],
                    ..object(
                        Some(ShapeDesc::Hull {
                            points: vec![
                                [0.0, 0.5, 0.0],
                                [0.5, -0.3, 0.3],
                                [-0.5, -0.3, 0.3],
                                [0.0, -0.3, -0.5],
                            ],
                        }),
                        [0.0, 1.0, 2.0],
                        dynamic,
                    )
                },
            ],
            joints: vec![
                JointSpec {
                    a: 0,
                    b: 1,
                    kind: JointKindSpec::BallSocket {
                        pivot: [0.0, 5.0, 0.0],
                        cone: Some(([0.0, -1.0, 0.0], 30.0)),
                    },
                    collide_connected: false,
                },
                JointSpec {
                    a: 3,
                    b: 2,
                    kind: JointKindSpec::Hinge {
                        pivot: [0.0, 1.0, 0.0],
                        axis: [0.0, 0.0, 1.0],
                        limits_deg: Some((-45.0, 45.0)),
                    },
                    collide_connected: true,
                },
                JointSpec {
                    a: 1,
                    b: 4,
                    kind: JointKindSpec::Distance {
                        anchor_a: [0.0, 4.0, 0.0],
                        anchor_b: [0.0, 1.0, 2.0],
                    },
                    collide_connected: false,
                },
            ],
        }
    }

    #[test]
    fn builds_every_shape_body_and_joint_kind() {
        let desc = sampler();
        let (world, objects) = build_world(&desc, &ModelRegistry::default()).unwrap();
        assert_eq!(objects.len(), desc.objects.len());
        assert_eq!(world.len(), 5 + desc.objects.len(), "pit planes + objects");
        assert_eq!(world.joints().count(), 3);
        let hull = world.body(objects[4].handle).unwrap();
        assert!(
            hull.orientation()
                .abs_diff_eq(desc.objects[4].orientation(), 1e-6)
        );
        assert_eq!(world.body(objects[0].handle).unwrap().collider(), None);
        assert_eq!(world.body(objects[3].handle).unwrap().inverse_mass(), 0.0);
    }

    #[test]
    fn failed_rebuild_leaves_the_world_alone_and_success_stales_handles() {
        let (mut world, objects) = build_world(&sampler(), &ModelRegistry::default()).unwrap();
        let mut bad = sampler();
        bad.objects[1].body = BodyType::Dynamic { mass: 0.0 };
        assert!(rebuild(&mut world, &bad, &ModelRegistry::default()).is_err());
        assert!(objects.iter().all(|o| world.body(o.handle).is_some()));
        rebuild(&mut world, &sampler(), &ModelRegistry::default()).unwrap();
        assert!(objects.iter().all(|o| world.body(o.handle).is_none()));
    }

    /// Stop after Play: rebuilding from the description restores the authored
    /// scene exactly, whatever the simulation did.
    #[test]
    fn rebuild_after_simulating_restores_the_authored_scene_bit_for_bit() {
        let desc = sampler();
        let state = |w: &World| -> Vec<[u32; 13]> {
            w.bodies()
                .map(|(_, b)| {
                    let mut s = [0; 13];
                    let values = [
                        b.position().to_array().as_slice(),
                        &b.orientation().to_array(),
                        &b.velocity().to_array(),
                        &b.angular_velocity().to_array(),
                    ]
                    .concat();
                    for (slot, v) in s.iter_mut().zip(values) {
                        *slot = v.to_bits();
                    }
                    s
                })
                .collect()
        };
        let (fresh, _) = build_world(&desc, &ModelRegistry::default()).unwrap();
        let (mut world, _) = build_world(&desc, &ModelRegistry::default()).unwrap();
        for _ in 0..120 {
            world.step(1.0 / 60.0);
        }
        assert_ne!(
            state(&world),
            state(&fresh),
            "the simulation should move things"
        );
        rebuild(&mut world, &desc, &ModelRegistry::default()).unwrap();
        assert_eq!(state(&world), state(&fresh));
    }

    #[test]
    fn file_stems_are_safe() {
        assert_eq!(file_stem("My Scene-2").as_deref(), Some("My_Scene-2"));
        assert_eq!(file_stem("../../etc/passwd").as_deref(), Some("etcpasswd"));
        assert_eq!(file_stem("C:\\x.ron").as_deref(), Some("Cxron"));
        assert_eq!(file_stem("  ").as_deref(), None);
        assert_eq!(file_stem("...").as_deref(), None);
        assert_eq!(file_stem(&"a".repeat(100)).map(|s| s.len()), Some(64));
    }

    #[test]
    fn oversized_files_are_refused_before_parsing() {
        let huge = " ".repeat(MAX_SCENE_BYTES + 1);
        assert!(matches!(from_ron(&huge), Err(SceneError::Parse(_))));
    }

    #[test]
    fn model_objects_round_trip_and_build_from_the_registry() {
        let mut desc = sampler();
        desc.objects.push(ObjectDesc {
            name: "barrel".into(),
            shape: Some(ShapeDesc::Model {
                name: "barrel".into(),
                scale: 1.5,
                collider: ModelCollider::Hull,
            }),
            position: [0.0, 2.0, -2.0],
            ..ObjectDesc::default()
        });
        assert_eq!(from_ron(&to_ron(&desc)).unwrap(), desc);
        let (world, objects) = build_world(&desc, &ModelRegistry::bundled()).unwrap();
        let barrel = objects.last().unwrap();
        assert!(barrel.model.is_some());
        let body = world.body(barrel.handle).unwrap();
        assert!(matches!(body.collider(), Some(Collider::Hull { .. })));
        assert_eq!(body.position(), Vec3::new(0.0, 2.0, -2.0));

        // Not in this registry: a specific error naming the model.
        let err = build_world(&desc, &ModelRegistry::default()).unwrap_err();
        assert_eq!(
            err,
            SceneError::Object {
                index: desc.objects.len() - 1,
                name: "barrel".into(),
                error: ObjectError::MissingModel("barrel".into()),
            }
        );
        assert!(err.to_string().contains("import it first"), "{err}");
    }

    #[test]
    fn hand_written_model_objects_default_to_hull_colliders() {
        let text = r#"(
            version: 1,
            name: "one barrel",
            arena: Floor,
            objects: [(shape: Some(Model(name: "barrel", scale: 1.0)), position: (0.0, 1.0, 0.0))],
        )"#;
        let desc = from_ron(text).unwrap();
        assert!(matches!(
            desc.objects[0].shape,
            Some(ShapeDesc::Model {
                collider: ModelCollider::Hull,
                ..
            })
        ));
    }

    #[test]
    fn surfaces_round_trip_and_older_files_get_the_defaults() {
        let mut desc = sampler();
        desc.objects[1].roughness = 0.2;
        desc.objects[1].metallic = 1.0;
        assert_eq!(from_ron(&to_ron(&desc)).unwrap(), desc);
        // A file from before M9b: no roughness or metallic fields.
        let old = r#"(version: 1, name: "old", arena: Floor,
            objects: [(shape: Some(Sphere(radius: 0.5)), position: (0.0, 1.0, 0.0))])"#;
        let o = &from_ron(old).unwrap().objects[0];
        assert_eq!((o.roughness, o.metallic), (0.6, 0.0));
        let (_, objects) = build_world(&desc, &ModelRegistry::default()).unwrap();
        assert_eq!((objects[1].roughness, objects[1].metallic), (0.2, 1.0));
    }

    #[test]
    fn ron_round_trips() {
        let desc = sampler();
        let text = to_ron(&desc);
        assert_eq!(from_ron(&text).unwrap(), desc);
    }

    #[test]
    fn hand_written_files_can_skip_defaults() {
        let text = r#"(
            version: 1,
            name: "tiny",
            arena: Floor,
            objects: [
                (shape: Some(Sphere(radius: 0.5)), position: (0.0, 2.0, 0.0)),
            ],
        )"#;
        let desc = from_ron(text).unwrap();
        assert!(desc.joints.is_empty());
        assert_eq!(desc.objects[0].body, BodyType::Dynamic { mass: 1.0 });
        let (world, _) = build_world(&desc, &ModelRegistry::default()).unwrap();
        assert_eq!(world.len(), 2, "floor + sphere");
    }

    #[test]
    fn euler_degrees_round_trip_through_quaternions() {
        let mut o = ObjectDesc {
            rotation_deg: [15.0, -40.0, 75.0],
            ..ObjectDesc::default()
        };
        let q = o.orientation();
        o.set_orientation(q);
        for (got, want) in o.rotation_deg.iter().zip([15.0, -40.0, 75.0]) {
            assert!((got - want).abs() < 1e-3, "{:?}", o.rotation_deg);
        }
    }

    #[test]
    fn bad_scenes_report_specific_errors() {
        assert!(matches!(
            from_ron("(version: 1,"),
            Err(SceneError::Parse(_))
        ));
        let mut desc = sampler();
        desc.version = 99;
        assert_eq!(
            from_ron(&to_ron(&desc)),
            Err(SceneError::UnsupportedVersion(99))
        );

        let err = |edit: fn(&mut SceneDesc)| {
            let mut desc = sampler();
            edit(&mut desc);
            build_world(&desc, &ModelRegistry::default()).expect_err("scene should be rejected")
        };
        assert_eq!(
            err(|d| d.objects[2].body = BodyType::Dynamic { mass: -1.0 }),
            SceneError::Object {
                index: 2,
                name: "thing".into(),
                error: ObjectError::Body(BodyError::InvalidMass),
            }
        );
        assert_eq!(
            err(|d| d.objects[1].shape = Some(ShapeDesc::Sphere { radius: 0.0 })),
            SceneError::Object {
                index: 1,
                name: "thing".into(),
                error: ObjectError::Body(BodyError::InvalidCollider),
            }
        );
        assert!(matches!(
            err(|d| d.objects[4].shape = Some(ShapeDesc::Hull {
                points: vec![[0.0; 3]; 3]
            })),
            SceneError::Object {
                index: 4,
                error: ObjectError::Hull(_),
                ..
            }
        ));
        assert_eq!(
            err(|d| d.joints[1].b = 42),
            SceneError::Joint {
                index: 1,
                error: JointSpecError::NoSuchObject(42),
            }
        );
        assert_eq!(
            err(|d| d.joints[0].b = 0),
            SceneError::Joint {
                index: 0,
                error: JointSpecError::Joint(JointError::SameBody),
            }
        );
        assert_eq!(
            err(|d| d.arena = Arena::Pit { half: 0.0 }),
            SceneError::InvalidArena
        );
        // Messages name the culprit.
        let message = err(|d| d.objects[2].body = BodyType::Dynamic { mass: -1.0 }).to_string();
        assert!(message.starts_with("object 2 'thing':"), "{message}");
    }
}
