//! The M2 demo scene: spheres of varied size and bounciness dropped into a
//! walled pit, plus how to draw it. Render sizes come from the physics colliders.

use glam::{Mat4, Vec3};
use std::collections::HashMap;

use impulse_physics::{BodyHandle, Collider, ConvexHull, HullId, Material, World};
use impulse_render::{DrawList, Instance, Light, LightKind, MeshData, MeshId, MeshKind, Vertex};

/// Renderer meshes registered for the hull shapes in the world.
pub type HullMeshes = HashMap<HullId, MeshId>;

use crate::models::{ModelMeshes, ModelRegistry, ModelVisual, fitting_scale};
use crate::pose::{Pose, interpolate};
use crate::scene_desc::{
    Arena, BodyType, LightDesc, LightKindDesc, ModelCollider, ObjectDesc, SCENE_VERSION, Scatter,
    SceneDesc, ShapeDesc, SkyPreset,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneObject {
    pub handle: BodyHandle,
    pub color: [f32; 4],
    /// Pose before the most recent physics step, for interpolation.
    pub prev: Pose,
    /// Model objects draw their mesh here instead of their collider.
    pub model: Option<ModelVisual>,
    /// Surface for built-in shapes (models use their own material).
    pub roughness: f32,
    pub metallic: f32,
    /// Glow strength (times the colour).
    pub emissive: f32,
    /// A light at the body's origin.
    pub light: Option<LightDesc>,
}

/// Spheres, boxes, and capsules in turn.
pub const BODY_COUNT: usize = 30;
/// Inner half-width of the pit (walls are planes at ±PIT).
const PIT: f32 = 6.0;
const RAIN_PIT: f32 = 14.0;
pub const RAIN_COUNT: usize = 1000;
const PALETTE_RAIN: [[f32; 4]; 5] = [
    [0.85, 0.35, 0.30, 1.0],
    [0.95, 0.70, 0.30, 1.0],
    [0.35, 0.65, 0.40, 1.0],
    [0.30, 0.50, 0.85, 1.0],
    [0.70, 0.45, 0.80, 1.0],
];
const WALL_HEIGHT: f32 = 2.0;
const WALL_THICKNESS: f32 = 0.2;
const FLOOR_SIZE: f32 = 40.0;
const FLOOR_COLOR: [f32; 4] = [0.32, 0.33, 0.35, 1.0];
const WALL_COLOR: [f32; 4] = [0.42, 0.47, 0.55, 1.0];
const GRID_COLOR: [f32; 4] = [0.12, 0.13, 0.15, 1.0];
const ROPE_COLOR: [f32; 4] = [0.85, 0.78, 0.60, 1.0];
const LOW_BOUNCE: [f32; 3] = [0.15, 0.40, 0.80];
const HIGH_BOUNCE: [f32; 3] = [0.95, 0.55, 0.15];
pub(crate) const STATIC_MATERIAL: Material = Material {
    restitution: 0.4,
    friction: 0.5,
};

/// Fractional part of `i · k`: a cheap, deterministic spread of values in [0, 1).
fn spread(i: usize, k: f32) -> f32 {
    (i as f32 * k).fract()
}

/// The floor plane and the four inward-facing wall planes: `(point, normal)`.
pub(crate) fn pit_planes(half: f32) -> [(Vec3, Vec3); 5] {
    [
        (Vec3::ZERO, Vec3::Y),
        (Vec3::new(-half, 0.0, 0.0), Vec3::X),
        (Vec3::new(half, 0.0, 0.0), Vec3::NEG_X),
        (Vec3::new(0.0, 0.0, -half), Vec3::Z),
        (Vec3::new(0.0, 0.0, half), Vec3::NEG_Z),
    ]
}

/// Inner half-width of the pit for a scene (Rain needs room for 1,000 bodies).
pub fn pit_half(demo: Demo) -> f32 {
    match demo {
        Demo::Rain => RAIN_PIT,
        _ => PIT,
    }
}

/// The demo scenes offered in the panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Demo {
    /// Spheres and boxes tumbling into the pit.
    Pit,
    /// Ten cubes stacked in a column (M4's stability showcase).
    Tower,
    /// Fifteen cubes in a five-row pyramid.
    Pyramid,
    /// 1,000 mixed bodies (M5's scale showcase).
    Rain,
    /// A chain, a rope bridge, and a ragdoll (M7's joints showcase).
    Joints,
    /// Bundled glTF models tumbling into the pit (M9's models showcase).
    Props,
    /// A roughness × metallic sphere chart and props (M9b's PBR showcase).
    Materials,
    /// Tinted glass and chain-link fences (R4's transparency showcase).
    Glass,
    /// Lamps, stage spots and glowing balls at dusk (R5's lights showcase).
    Lights,
    /// 10,000 scattered props on a wide floor (R7's culling and LOD showcase).
    Field,
}

impl Demo {
    pub const ALL: [Demo; 10] = [
        Demo::Pit,
        Demo::Tower,
        Demo::Pyramid,
        Demo::Rain,
        Demo::Joints,
        Demo::Props,
        Demo::Materials,
        Demo::Glass,
        Demo::Lights,
        Demo::Field,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Demo::Pit => "Pit",
            Demo::Tower => "Tower",
            Demo::Pyramid => "Pyramid",
            Demo::Rain => "Rain",
            Demo::Joints => "Joints",
            Demo::Props => "Props",
            Demo::Materials => "Materials",
            Demo::Glass => "Glass",
            Demo::Lights => "Lights",
            Demo::Field => "Field",
        }
    }

    /// Parses a scene name, e.g. from `?scene=rain` (case-insensitive).
    pub fn from_name(name: &str) -> Option<Demo> {
        Demo::ALL
            .into_iter()
            .find(|d| d.label().eq_ignore_ascii_case(name))
    }
}

pub const TOWER_HEIGHT: usize = 10;
pub const PYRAMID_BASE: usize = 5;
const STACK_COLORS: [[f32; 4]; 3] = [
    [0.80, 0.35, 0.25, 1.0],
    [0.90, 0.75, 0.40, 1.0],
    [0.30, 0.55, 0.75, 1.0],
];

/// Half-size of the floor object a new scene starts with (m).
pub const NEW_FLOOR_HALF: f32 = FLOOR_SIZE / 2.0;

/// A blank scene: no surroundings, just a finite floor object (static box,
/// top at y = 0) that can be edited, resized, or deleted like anything else.
pub fn empty_desc() -> SceneDesc {
    let [r, g, b, _] = FLOOR_COLOR;
    SceneDesc {
        version: SCENE_VERSION,
        name: "untitled".to_string(),
        arena: Arena::Empty,
        sky: SkyPreset::Day,
        scatter: None,
        objects: vec![ObjectDesc {
            name: "floor".to_string(),
            shape: Some(ShapeDesc::Box {
                half_extents: [NEW_FLOOR_HALF, 0.5, NEW_FLOOR_HALF],
            }),
            position: [0.0, -0.5, 0.0],
            body: BodyType::Static,
            restitution: STATIC_MATERIAL.restitution,
            friction: STATIC_MATERIAL.friction,
            color: [r, g, b],
            ..ObjectDesc::default()
        }],
        joints: Vec::new(),
    }
}

/// Default pit half-width when switching an open floor to a pit.
pub const DEFAULT_PIT: f32 = PIT;

/// A demo as a scene description. Deterministic.
pub fn demo_desc(demo: Demo) -> SceneDesc {
    let mut desc = demo_world_desc(demo);
    crate::scene_desc::localize_joints(&mut desc);
    desc.version = SCENE_VERSION;
    if demo == Demo::Joints {
        // Grouped after the joints are localized: nothing moves.
        let (_, _, parents) = crate::joint_scene::joint_objects();
        for (child, parent) in parents {
            crate::scene_desc::set_parent_keep_world(&mut desc, child, Some(parent));
        }
    }
    desc
}

/// A demo with world-space joints, i.e. the v1 format (the generators author
/// joints in world space).
pub fn demo_world_desc(demo: Demo) -> SceneDesc {
    let (objects, joints) = match demo {
        Demo::Pit => (pit_objects(), Vec::new()),
        Demo::Tower => (
            (0..TOWER_HEIGHT)
                .map(|i| stack_cube(Vec3::new(0.0, 0.5 + i as f32, 0.0), i))
                .collect(),
            Vec::new(),
        ),
        Demo::Pyramid => {
            let mut objects = Vec::new();
            for row in 0..PYRAMID_BASE {
                let count = PYRAMID_BASE - row;
                for i in 0..count {
                    let x = (i as f32 - (count - 1) as f32 / 2.0) * 1.025;
                    objects.push(stack_cube(Vec3::new(x, 0.5 + row as f32, 0.0), row));
                }
            }
            (objects, Vec::new())
        }
        Demo::Rain => (rain_objects(), Vec::new()),
        Demo::Joints => {
            let (objects, joints, _) = crate::joint_scene::joint_objects();
            (objects, joints)
        }
        Demo::Props => (props_objects(), Vec::new()),
        Demo::Materials => (materials_objects(), Vec::new()),
        Demo::Glass => (glass_objects(), Vec::new()),
        Demo::Lights => (lights_objects(), Vec::new()),
        Demo::Field => (field_objects(), Vec::new()),
    };
    let mut desc = SceneDesc {
        version: 1,
        name: demo.label().to_string(),
        // The chart is for looking at: no walls in the way.
        arena: match demo {
            Demo::Materials | Demo::Glass | Demo::Lights => Arena::Floor,
            // Its own floor object, as wide as the props.
            Demo::Field => Arena::Empty,
            _ => Arena::Pit {
                half: pit_half(demo),
            },
        },
        sky: match demo {
            Demo::Lights => SkyPreset::Dusk,
            // Metals reflecting a real sky.
            Demo::Materials | Demo::Field => SkyPreset::Clouds,
            _ => SkyPreset::Day,
        },
        scatter: (demo == Demo::Field).then_some(FIELD_SCATTER),
        objects,
        joints,
    };
    for (i, object) in desc.objects.iter_mut().enumerate() {
        if object.name.is_empty() {
            object.name = default_name(object, i);
        }
    }
    desc
}

/// "box 3", "anchor 0": a readable name for unnamed objects.
pub fn default_name(object: &ObjectDesc, index: usize) -> String {
    let kind = match &object.shape {
        None => "empty",
        Some(ShapeDesc::Sphere { .. }) => "sphere",
        Some(ShapeDesc::Box { .. }) => "box",
        Some(ShapeDesc::Capsule { .. }) => "capsule",
        Some(ShapeDesc::Hull { .. }) => "gem",
        Some(ShapeDesc::Model { name, .. }) => name.as_str(),
    };
    format!("{kind} {index}")
}

/// Replaces everything in `world` with `demo`, keeping its solver settings.
#[cfg(test)]
pub fn build(world: &mut World, demo: Demo) -> Vec<SceneObject> {
    crate::scene_desc::rebuild(
        world,
        &demo_desc(demo),
        &crate::models::ModelRegistry::bundled(),
    )
    .expect("demo scenes are valid")
}

/// A demo object from physics-style values.
#[allow(clippy::too_many_arguments)]
fn demo_object(
    shape: ShapeDesc,
    position: Vec3,
    orientation: glam::Quat,
    mass: f32,
    material: Material,
    color: [f32; 4],
    velocity: Vec3,
    spin: Vec3,
) -> ObjectDesc {
    let mut object = ObjectDesc {
        shape: Some(shape),
        position: position.to_array(),
        body: BodyType::Dynamic { mass },
        restitution: material.restitution,
        friction: material.friction,
        color: [color[0], color[1], color[2]],
        velocity: velocity.to_array(),
        angular_velocity: spin.to_array(),
        ..ObjectDesc::default()
    };
    object.set_orientation(orientation);
    object
}

/// Sphere chart columns (roughness) and rows (metallic).
pub const CHART_COLUMNS: usize = 6;
pub const CHART_ROWS: usize = 3;

/// The classic material chart: roughness rises left to right, metallic front to
/// back, all one warm colour; static so it stays put. Props sit beside it.
fn materials_objects() -> Vec<ObjectDesc> {
    let mut objects = Vec::new();
    for row in 0..CHART_ROWS {
        for column in 0..CHART_COLUMNS {
            let roughness = 0.05 + 0.95 * column as f32 / (CHART_COLUMNS - 1) as f32;
            let metallic = row as f32 / (CHART_ROWS - 1) as f32;
            objects.push(ObjectDesc {
                name: format!("r{roughness:.2} m{metallic:.1}"),
                shape: Some(ShapeDesc::Sphere { radius: 0.4 }),
                position: [column as f32 * 1.1 - 2.75, 0.4, row as f32 * 1.1 - 3.0],
                body: BodyType::Static,
                color: [0.95, 0.64, 0.38],
                roughness,
                metallic,
                ..ObjectDesc::default()
            });
        }
    }
    let registry = crate::models::ModelRegistry::bundled();
    for (i, name) in ["barrel", "chest", "rock-a"].iter().enumerate() {
        let id = registry.id(name).expect("bundled");
        let scale = crate::models::fitting_scale(registry.get(id).expect("bundled"), 1.0);
        objects.push(ObjectDesc {
            shape: Some(ShapeDesc::Model {
                name: name.to_string(),
                scale,
                collider: ModelCollider::Hull,
            }),
            position: [i as f32 * 1.8 - 1.8, 0.8, 1.8],
            color: [1.0; 3],
            ..ObjectDesc::default()
        });
    }
    objects
}

/// Glass tints: blue, green, amber, rose.
const GLASS_TINTS: [[f32; 3]; 4] = [
    [0.55, 0.75, 1.0],
    [0.6, 1.0, 0.7],
    [1.0, 0.8, 0.45],
    [1.0, 0.6, 0.7],
];

/// Panes and blocks of tinted glass in front of opaque shapes, inside a ring
/// of chain-link fences: blending, sorting and alpha-tested cut-outs.
fn glass_objects() -> Vec<ObjectDesc> {
    let mut objects = Vec::new();
    // Opaque things to look at through the glass.
    for (i, x) in [-3.0_f32, 0.0, 3.0].into_iter().enumerate() {
        let [r, g, b, _] = STACK_COLORS[i];
        objects.push(ObjectDesc {
            shape: Some(ShapeDesc::Sphere { radius: 0.6 }),
            position: [x, 0.6, -2.5],
            body: BodyType::Static,
            color: [r, g, b],
            ..ObjectDesc::default()
        });
    }
    // Upright panes, overlapping in depth so sorting shows.
    for (i, &tint) in GLASS_TINTS.iter().enumerate() {
        objects.push(ObjectDesc {
            shape: Some(ShapeDesc::Box {
                half_extents: [0.9, 1.0, 0.05],
            }),
            position: [i as f32 * 1.4 - 2.1, 1.0, -0.6 + (i % 2) as f32 * 0.8],
            body: BodyType::Static,
            color: tint,
            opacity: 0.35,
            roughness: 0.05,
            ..ObjectDesc::default()
        });
    }
    // Glass blocks and balls that fall and stack.
    for i in 0..8 {
        let shape = if i % 2 == 0 {
            ShapeDesc::Box {
                half_extents: [0.35; 3],
            }
        } else {
            ShapeDesc::Sphere { radius: 0.35 }
        };
        objects.push(ObjectDesc {
            shape: Some(shape),
            position: [
                (i % 4) as f32 * 1.0 - 1.5 + spread(i, 0.37) * 0.3,
                2.5 + (i / 4) as f32 * 1.2,
                1.6 + spread(i, 0.53) * 0.4,
            ],
            body: BodyType::Dynamic { mass: 1.0 },
            color: GLASS_TINTS[i % GLASS_TINTS.len()],
            opacity: 0.45,
            roughness: 0.1,
            ..ObjectDesc::default()
        });
    }
    // A ring of fence panels around it all.
    for i in 0..16 {
        let angle = i as f32 / 16.0 * std::f32::consts::TAU;
        let radius = 6.0;
        let mut fence = ObjectDesc {
            shape: Some(ShapeDesc::Model {
                name: "fence".to_string(),
                scale: 1.0,
                collider: ModelCollider::Box,
            }),
            position: [angle.cos() * radius, 0.0, angle.sin() * radius],
            body: BodyType::Static,
            color: [1.0; 3],
            ..ObjectDesc::default()
        };
        fence.set_orientation(glam::Quat::from_rotation_y(
            std::f32::consts::FRAC_PI_2 - angle,
        ));
        objects.push(fence);
    }
    objects
}

/// Lamp colours: warm white, amber, teal, violet.
const LAMP_COLORS: [[f32; 3]; 4] = [
    [1.0, 0.82, 0.6],
    [1.0, 0.55, 0.2],
    [0.3, 0.9, 0.85],
    [0.7, 0.45, 1.0],
];

/// A colonnade of coloured lamps, two shadowed stage spots over a pile of
/// blocks, glowing balls that roll, and a field of 64 small lanterns: every
/// light kind, moving lights, and enough of them for clustering to matter.
fn lights_objects() -> Vec<ObjectDesc> {
    let mut objects = Vec::new();
    let lamp = |position: [f32; 3], radius: f32, color: [f32; 3], light: LightDesc| ObjectDesc {
        shape: Some(ShapeDesc::Sphere { radius }),
        position,
        body: BodyType::Static,
        color,
        emissive: 4.0,
        light: Some(LightDesc { color, ..light }),
        ..ObjectDesc::default()
    };
    // Pillars down both sides, a lamp on the inside of each.
    for side in [-1.0_f32, 1.0] {
        for i in 0..6 {
            let z = i as f32 * 2.4 - 6.0;
            objects.push(ObjectDesc {
                shape: Some(ShapeDesc::Box {
                    half_extents: [0.25, 1.5, 0.25],
                }),
                position: [side * 3.5, 1.5, z],
                body: BodyType::Static,
                color: [0.75, 0.72, 0.68],
                ..ObjectDesc::default()
            });
            objects.push(lamp(
                [side * 3.1, 2.3, z],
                0.1,
                LAMP_COLORS[(i + side as usize) % LAMP_COLORS.len()],
                LightDesc {
                    intensity: 5.0,
                    range: 7.0,
                    ..LightDesc::default()
                },
            ));
        }
    }
    // Two stage spots high up, crossing over the middle.
    for (side, color) in [(-1.0_f32, [1.0, 0.92, 0.82]), (1.0, [0.6, 0.75, 1.0])] {
        let mut spot = ObjectDesc {
            name: "stage spot".to_string(),
            position: [side * 2.0, 6.5, 0.0],
            body: BodyType::Static,
            light: Some(LightDesc {
                kind: LightKindDesc::Spot {
                    inner_deg: 16.0,
                    outer_deg: 24.0,
                },
                color,
                intensity: 70.0,
                range: 14.0,
                shadows: true,
            }),
            ..ObjectDesc::default()
        };
        spot.rotation_deg = [0.0, 0.0, side * 18.0];
        objects.push(spot);
    }
    // A pile of pale blocks for the spots to light and shadow.
    for i in 0..10 {
        objects.push(ObjectDesc {
            shape: Some(ShapeDesc::Box {
                half_extents: [0.35; 3],
            }),
            position: [
                spread(i, 0.37) * 1.6 - 0.8,
                0.4 + (i / 3) as f32 * 0.75,
                spread(i, 0.53) * 1.6 - 0.8,
            ],
            body: BodyType::Dynamic { mass: 1.0 },
            color: [0.8, 0.8, 0.78],
            ..ObjectDesc::default()
        });
    }
    // Glowing balls dropped down the aisle, rolling as they land.
    for i in 0..6 {
        let color = LAMP_COLORS[(i + 1) % LAMP_COLORS.len()];
        objects.push(ObjectDesc {
            shape: Some(ShapeDesc::Sphere { radius: 0.2 }),
            position: [
                spread(i, 0.71) * 3.0 - 1.5,
                3.0 + i as f32 * 0.5,
                i as f32 * 1.8 - 4.5,
            ],
            body: BodyType::Dynamic { mass: 0.5 },
            color,
            emissive: 4.0,
            velocity: [
                spread(i, 0.29) * 4.0 - 2.0,
                0.0,
                spread(i, 0.61) * 4.0 - 2.0,
            ],
            light: Some(LightDesc {
                color,
                intensity: 2.5,
                range: 4.0,
                ..LightDesc::default()
            }),
            ..ObjectDesc::default()
        });
    }
    // A field of small lanterns behind the colonnade.
    for i in 0..64 {
        let (x, z) = ((i % 8) as f32, (i / 8) as f32);
        let hue = spread(i, 0.618);
        let color = [
            0.55 + 0.45 * (hue * std::f32::consts::TAU).cos(),
            0.55 + 0.45 * ((hue + 0.33) * std::f32::consts::TAU).cos(),
            0.55 + 0.45 * ((hue + 0.67) * std::f32::consts::TAU).cos(),
        ];
        objects.push(lamp(
            [x * 1.4 - 4.9, 0.3, -9.0 - z * 1.4],
            0.06,
            color,
            LightDesc {
                intensity: 0.8,
                range: 2.2,
                ..LightDesc::default()
            },
        ));
    }
    objects
}

/// Models dropped into the pit
/// (all bundled, so the scene always resolves).
pub const PROPS: [&str; 8] = [
    "barrel",
    "box",
    "box-large",
    "bucket",
    "chest",
    "rock-a",
    "rock-b",
    "tree-log",
];
pub const PROP_COUNT: usize = 24;

fn props_objects() -> Vec<ObjectDesc> {
    let registry = crate::models::ModelRegistry::bundled();
    (0..PROP_COUNT)
        .map(|i| {
            let name = PROPS[i % PROPS.len()];
            let id = registry.id(name).expect("bundled");
            let scale = crate::models::fitting_scale(registry.get(id).expect("bundled"), 1.2);
            let position = Vec3::new(
                (i % 4) as f32 * 2.4 - 3.6,
                2.0 + (i / 8) as f32 * 1.6 + spread(i, 0.37),
                ((i / 4) % 2) as f32 * 2.4 - 1.2 + ((i / 8) as f32 - 1.0) * 1.5,
            );
            let mut object = ObjectDesc {
                shape: Some(ShapeDesc::Model {
                    name: name.to_string(),
                    scale,
                    collider: ModelCollider::Hull,
                }),
                position: position.to_array(),
                body: BodyType::Dynamic { mass: 2.0 },
                color: [1.0; 3],
                angular_velocity: [
                    spread(i, 0.31) - 0.5,
                    spread(i, 0.47) - 0.5,
                    spread(i, 0.71) - 0.5,
                ],
                ..ObjectDesc::default()
            };
            object.set_orientation(glam::Quat::from_rotation_y(
                spread(i, 0.53) * std::f32::consts::TAU,
            ));
            object
        })
        .collect()
}

/// Layers of mixed spheres and boxes filling the Rain pit (same recipe as the
/// physics benchmark, so browser and native numbers are comparable).
fn rain_objects() -> Vec<ObjectDesc> {
    let per_row = ((2.0 * RAIN_PIT - 2.0) / 1.2).floor() as usize;
    (0..RAIN_COUNT)
        .map(|i| {
            let (col, row, layer) = (
                i % per_row,
                (i / per_row) % per_row,
                i / (per_row * per_row),
            );
            let position = Vec3::new(
                -RAIN_PIT + 1.0 + col as f32 * 1.2,
                1.0 + layer as f32 * 1.2,
                -RAIN_PIT + 1.0 + row as f32 * 1.2,
            );
            let size = 0.25 + 0.2 * spread(i, 0.37);
            let shape = if i % 2 == 0 {
                ShapeDesc::Sphere { radius: size }
            } else {
                ShapeDesc::Box {
                    half_extents: [size, size * 0.8, size * 0.9],
                }
            };
            demo_object(
                shape,
                position,
                glam::Quat::from_rotation_y(spread(i, 0.53) * 6.0),
                1.0,
                Material {
                    restitution: 0.2,
                    friction: 0.5,
                },
                PALETTE_RAIN[i % PALETTE_RAIN.len()],
                Vec3::ZERO,
                Vec3::ZERO,
            )
        })
        .collect()
}

/// The pit demo (tests use it as the canonical scene).
/// A faceted gem: points on a slightly squashed sphere (golden-angle spiral).
fn gem_points(seed: usize, radius: f32) -> Vec<Vec3> {
    let squash = 0.7 + 0.5 * spread(seed, 0.19);
    (0..10)
        .map(|k| {
            let y = 1.0 - 2.0 * (k as f32 + 0.5) / 10.0;
            let ring = (1.0 - y * y).sqrt();
            let phi = k as f32 * 2.399_963 + seed as f32;
            Vec3::new(ring * phi.cos(), y * squash, ring * phi.sin()) * radius
        })
        .collect()
}

/// Flat-shaded mesh of a hull: each face fanned into triangles carrying its normal.
pub fn hull_mesh(hull: &ConvexHull) -> MeshData {
    let mut mesh = MeshData {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    for face in hull.faces() {
        let base = mesh.vertices.len() as u32;
        mesh.vertices.extend(face.vertices.iter().map(|&v| Vertex {
            position: hull.vertices()[v].to_array(),
            normal: face.normal.to_array(),
            uv: [0.0; 2],
            tangent: impulse_render::perpendicular_tangent(face.normal.to_array()),
        }));
        for k in 1..face.vertices.len() as u32 - 1 {
            mesh.indices.extend([base, base + k, base + k + 1]);
        }
    }
    mesh
}

/// Distinct hull shapes used by bodies in the world.
pub fn hulls_in_world(world: &World) -> Vec<HullId> {
    let mut ids = Vec::new();
    for (_, body) in world.bodies() {
        if let Some(Collider::Hull { hull }) = body.collider()
            && !ids.contains(&hull)
        {
            ids.push(hull);
        }
    }
    ids
}

#[cfg(test)]
pub fn demo_scene(world: &mut World) -> Vec<SceneObject> {
    build(world, Demo::Pit)
}

fn stack_cube(position: Vec3, layer: usize) -> ObjectDesc {
    demo_object(
        ShapeDesc::Box {
            half_extents: [0.5; 3],
        },
        position,
        glam::Quat::IDENTITY,
        1.0,
        Material {
            restitution: 0.0,
            friction: 0.6,
        },
        STACK_COLORS[layer % STACK_COLORS.len()],
        Vec3::ZERO,
        Vec3::ZERO,
    )
}

/// The Field demo's props.
pub const FIELD_SCATTER: Scatter = Scatter {
    count: 10_000,
    half_extent: 100.0,
    seed: 1,
};

/// A floor reaching past the scattered props, and a short tower to knock.
fn field_objects() -> Vec<ObjectDesc> {
    let [r, g, b, _] = FLOOR_COLOR;
    let half = FIELD_SCATTER.half_extent + 10.0;
    let mut objects = vec![ObjectDesc {
        name: "floor".to_string(),
        shape: Some(ShapeDesc::Box {
            half_extents: [half, 0.5, half],
        }),
        position: [0.0, -0.5, 0.0],
        body: BodyType::Static,
        restitution: STATIC_MATERIAL.restitution,
        friction: STATIC_MATERIAL.friction,
        color: [r, g, b],
        ..ObjectDesc::default()
    }];
    objects.extend((0..5).map(|i| stack_cube(Vec3::new(0.0, 0.5 + i as f32, 0.0), i)));
    objects
}

/// Bundled models a scatter places, and their sizes (m).
const SCATTER_MODELS: [(&str, f32); 6] = [
    ("rock-a", 1.2),
    ("rock-b", 1.6),
    ("tree-log", 1.8),
    ("barrel", 0.9),
    ("box", 0.7),
    ("bucket", 0.45),
];
/// Earthy colours for the plain spheres and boxes.
const SCATTER_COLORS: [[f32; 4]; 5] = [
    [0.55, 0.50, 0.45, 1.0],
    [0.40, 0.45, 0.35, 1.0],
    [0.62, 0.55, 0.40, 1.0],
    [0.35, 0.38, 0.42, 1.0],
    [0.70, 0.45, 0.30, 1.0],
];

/// The instances a [`Scatter`] draws: `count` props (bundled models, plain
/// spheres and boxes) at seeded random spots, sizes and headings on y = 0,
/// clear of the middle. Deterministic. Models that aren't uploaded are left
/// out (plain shapes take their place); a model prop is one instance per
/// mesh.
pub fn scatter_instances(
    scatter: &Scatter,
    models: &ModelRegistry,
    meshes: &ModelMeshes,
) -> Vec<Instance> {
    // xorshift64*, in [0, 1).
    let mut state = (scatter.seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32
    };
    let kinds: Vec<(&Vec<MeshId>, f32, f32)> = SCATTER_MODELS
        .iter()
        .filter_map(|&(name, size)| {
            let id = models.id(name)?;
            let asset = models.get(id)?;
            let scale = fitting_scale(asset, size);
            Some((meshes.get(&id)?, scale, -asset.data.min.y))
        })
        .collect();
    let half = scatter.half_extent;
    let clear = (half * 0.5).min(6.0);
    let mut instances = Vec::with_capacity(scatter.count as usize * 5 / 4);
    let mut placed = 0;
    while placed < scatter.count {
        let (x, z) = ((next() * 2.0 - 1.0) * half, (next() * 2.0 - 1.0) * half);
        if x * x + z * z < clear * clear {
            continue;
        }
        placed += 1;
        let ground = Vec3::new(x, 0.0, z);
        let heading = glam::Quat::from_rotation_y(next() * std::f32::consts::TAU);
        let pick = next();
        let jitter = 0.7 + 0.6 * next();
        let color = SCATTER_COLORS[(next() * SCATTER_COLORS.len() as f32) as usize];
        let roughness = 0.3 + 0.6 * next();
        let plain = |mesh: MeshKind, size: f32| Instance {
            mesh: mesh.into(),
            transform: Mat4::from_scale_rotation_translation(
                Vec3::splat(size),
                heading,
                ground + Vec3::Y * size * 0.5,
            ),
            color,
            roughness,
            metallic: 0.0,
            emissive: 0.0,
        };
        if pick < 0.2 || kinds.is_empty() {
            instances.push(plain(MeshKind::Sphere, 0.6 * jitter + 0.2));
        } else if pick < 0.3 {
            instances.push(plain(MeshKind::Cube, 0.5 * jitter + 0.2));
        } else {
            let (ids, scale, lift) = kinds[(next() * kinds.len() as f32) as usize % kinds.len()];
            let scale = scale * jitter;
            let transform = Mat4::from_scale_rotation_translation(
                Vec3::splat(scale),
                heading,
                ground + Vec3::Y * lift * scale,
            );
            // 1, 1: the model's own material decides.
            instances.extend(ids.iter().map(|&mesh| Instance {
                mesh,
                transform,
                color: [1.0; 4],
                roughness: 1.0,
                metallic: 1.0,
                emissive: 0.0,
            }));
        }
    }
    instances
}

fn pit_objects() -> Vec<ObjectDesc> {
    (0..BODY_COUNT)
        .map(|i| {
            let radius = 0.3 + 0.4 * spread(i, 0.37);
            let restitution = 0.1 + 0.8 * spread(i, 0.61);
            let position = Vec3::new(
                (i % 5) as f32 * 2.0 - 4.0,
                3.0 + i as f32 * 0.4,
                (i / 5) as f32 * 1.6 - 4.0,
            );
            let velocity = Vec3::new(
                spread(i, 0.23) * 4.0 - 2.0,
                0.0,
                spread(i, 0.79) * 4.0 - 2.0,
            );
            let tumble = Vec3::new(
                spread(i, 0.31) - 0.5,
                spread(i, 0.47) - 0.5,
                spread(i, 0.71) - 0.5,
            ) * 4.0;
            // Cycle sphere, box, capsule, gem (convex hull); all but spheres spawn
            // tilted and spinning.
            let (shape, mass, orientation, spin) = if i % 4 == 0 {
                (
                    ShapeDesc::Sphere { radius },
                    8.0 * radius.powi(3),
                    glam::Quat::IDENTITY,
                    Vec3::ZERO,
                )
            } else if i % 4 == 3 {
                let points = gem_points(i, radius * 1.1)
                    .into_iter()
                    .map(|p| p.to_array())
                    .collect();
                (
                    ShapeDesc::Hull { points },
                    8.0 * radius.powi(3),
                    glam::Quat::from_axis_angle(Vec3::Z, spread(i, 0.53) * std::f32::consts::TAU),
                    tumble,
                )
            } else if i % 4 == 2 {
                (
                    ShapeDesc::Capsule {
                        half_height: radius * (1.0 + spread(i, 0.43)),
                        radius: radius * 0.6,
                    },
                    8.0 * radius.powi(3),
                    glam::Quat::from_axis_angle(Vec3::X, spread(i, 0.53) * std::f32::consts::TAU),
                    tumble,
                )
            } else {
                let half_extents =
                    Vec3::new(radius, radius * (0.6 + 0.6 * spread(i, 0.43)), radius * 0.8);
                let axis = Vec3::new(spread(i, 0.17) - 0.5, 1.0, spread(i, 0.29) - 0.5).normalize();
                (
                    ShapeDesc::Box {
                        half_extents: half_extents.to_array(),
                    },
                    8.0 * half_extents.x * half_extents.y * half_extents.z,
                    glam::Quat::from_axis_angle(axis, spread(i, 0.53) * std::f32::consts::TAU),
                    tumble,
                )
            };
            let t = (restitution - 0.1) / 0.8;
            let c = |k: usize| LOW_BOUNCE[k] + (HIGH_BOUNCE[k] - LOW_BOUNCE[k]) * t;
            demo_object(
                shape,
                position,
                orientation,
                mass,
                Material {
                    restitution,
                    friction: 0.4,
                },
                [c(0), c(1), c(2), 1.0],
                velocity,
                spin,
            )
        })
        .collect()
}

pub fn current_pose(world: &World, handle: BodyHandle) -> Option<Pose> {
    world.body(handle).map(|b| Pose {
        position: b.position(),
        orientation: b.orientation(),
    })
}

/// Records each body's pose; call right before stepping physics.
pub fn snapshot(world: &World, objects: &mut [SceneObject]) {
    for object in objects {
        if let Some(pose) = current_pose(world, object.handle) {
            object.prev = pose;
        }
    }
}

/// True once every body has fallen below `y` (a safety net: the demo re-drops them).
pub fn all_below(world: &World, objects: &[SceneObject], y: f32) -> bool {
    objects
        .iter()
        .filter_map(|o| current_pose(world, o.handle))
        .all(|pose| pose.position.y < y)
}

/// Meshes (with transforms in body space) that draw a collider; empty for planes
/// (drawn as scenery). A capsule is a cylinder plus a sphere at each end.
fn collider_visual(collider: Collider, hull_meshes: &HullMeshes) -> Vec<(MeshId, Mat4)> {
    let kind = |k: MeshKind, m: Mat4| (MeshId::from(k), m);
    match collider {
        Collider::Sphere { radius } => vec![kind(
            MeshKind::Sphere,
            Mat4::from_scale(Vec3::splat(radius * 2.0)),
        )],
        Collider::Box { half_extents } => {
            vec![kind(MeshKind::Cube, Mat4::from_scale(half_extents * 2.0))]
        }
        Collider::Capsule {
            half_height,
            radius,
        } => {
            let cap = |y: f32| {
                Mat4::from_translation(Vec3::Y * y) * Mat4::from_scale(Vec3::splat(radius * 2.0))
            };
            vec![
                kind(
                    MeshKind::Cylinder,
                    Mat4::from_scale(Vec3::new(radius * 2.0, half_height * 2.0, radius * 2.0)),
                ),
                kind(MeshKind::Sphere, cap(half_height)),
                kind(MeshKind::Sphere, cap(-half_height)),
            ]
        }
        // Hull vertices are already in body space, at true size.
        Collider::Hull { hull } => hull_meshes
            .get(&hull)
            .map(|&mesh| vec![(mesh, Mat4::IDENTITY)])
            .unwrap_or_default(),
        Collider::Plane { .. } => Vec::new(),
    }
}

/// Rough size for drawing a body's axes.
fn collider_extent(collider: Collider) -> f32 {
    match collider {
        Collider::Sphere { radius } => radius * 2.0,
        Collider::Box { half_extents } => half_extents.max_element() * 2.0,
        Collider::Capsule {
            half_height,
            radius,
        } => (half_height + radius) * 2.0,
        Collider::Hull { .. } => 1.0,
        Collider::Plane { .. } => 0.0,
    }
}

fn push_scenery(list: &mut DrawList, arena: Arena) {
    if arena == Arena::Empty {
        return;
    }
    list.push_instance(
        MeshKind::Plane,
        Mat4::from_scale(Vec3::new(FLOOR_SIZE, 1.0, FLOOR_SIZE)),
        FLOOR_COLOR,
    );
    let Arena::Pit { half: pit } = arena else {
        return;
    };
    // Thin boxes just outside each wall plane, so their inner faces sit on it.
    let length = 2.0 * (pit + WALL_THICKNESS);
    let offset = pit + WALL_THICKNESS / 2.0;
    let y = WALL_HEIGHT / 2.0;
    for (center, size) in [
        (
            Vec3::new(-offset, y, 0.0),
            Vec3::new(WALL_THICKNESS, WALL_HEIGHT, length),
        ),
        (
            Vec3::new(offset, y, 0.0),
            Vec3::new(WALL_THICKNESS, WALL_HEIGHT, length),
        ),
        (
            Vec3::new(0.0, y, -offset),
            Vec3::new(length, WALL_HEIGHT, WALL_THICKNESS),
        ),
        (
            Vec3::new(0.0, y, offset),
            Vec3::new(length, WALL_HEIGHT, WALL_THICKNESS),
        ),
    ] {
        list.push_instance(
            MeshKind::Cube,
            Mat4::from_scale_rotation_translation(size, glam::Quat::IDENTITY, center),
            WALL_COLOR,
        );
    }
}

/// How far a selected object's color moves toward white.
const SELECTED_TINT: f32 = 0.35;
const OUTLINE_COLOR: [f32; 4] = [1.0, 0.85, 0.2, 1.0];
const DESCENDANT_OUTLINE: [f32; 4] = [0.75, 0.6, 0.25, 1.0];

/// Half extents of a collider's bounding box in body space.
fn local_half_extents(collider: Collider, world: &World) -> Vec3 {
    match collider {
        Collider::Sphere { radius } => Vec3::splat(radius),
        Collider::Box { half_extents } => half_extents,
        Collider::Capsule {
            half_height,
            radius,
        } => Vec3::new(radius, half_height + radius, radius),
        Collider::Hull { hull } => world.hull(hull).map_or(Vec3::splat(0.5), |h| {
            h.vertices().iter().fold(Vec3::ZERO, |m, v| m.max(v.abs()))
        }),
        Collider::Plane { .. } => Vec3::ZERO,
    }
}

const COLLIDER_COLOR: [f32; 4] = [0.2, 0.95, 0.6, 1.0];

/// A collider's outline: hull edges, a box, or circles for round shapes.
fn push_collider_wireframe(
    list: &mut DrawList,
    world: &World,
    collider: Collider,
    transform: Mat4,
) {
    let circle = |list: &mut DrawList, center: Vec3, radius: f32, axis: usize| {
        let point = |k: usize| {
            let a = k as f32 / 24.0 * std::f32::consts::TAU;
            let (s, c) = a.sin_cos();
            let local = match axis {
                0 => Vec3::new(0.0, c, s),
                1 => Vec3::new(c, 0.0, s),
                _ => Vec3::new(c, s, 0.0),
            };
            transform.transform_point3(center + local * radius)
        };
        for k in 0..24 {
            list.push_line(point(k), point(k + 1), COLLIDER_COLOR);
        }
    };
    match collider {
        Collider::Hull { hull } => {
            let Some(hull) = world.hull(hull) else { return };
            for face in hull.faces() {
                let n = face.vertices.len();
                for k in 0..n {
                    let a = hull.vertices()[face.vertices[k]];
                    let b = hull.vertices()[face.vertices[(k + 1) % n]];
                    list.push_line(
                        transform.transform_point3(a),
                        transform.transform_point3(b),
                        COLLIDER_COLOR,
                    );
                }
            }
        }
        Collider::Box { half_extents } => {
            push_box_outline_colored(list, transform, half_extents, COLLIDER_COLOR);
        }
        Collider::Sphere { radius } => {
            for axis in 0..3 {
                circle(list, Vec3::ZERO, radius, axis);
            }
        }
        Collider::Capsule {
            half_height,
            radius,
        } => {
            for y in [-half_height, half_height] {
                circle(list, Vec3::Y * y, radius, 1);
            }
            for (x, z) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                let side = Vec3::new(x, 0.0, z) * radius;
                list.push_line(
                    transform.transform_point3(side - Vec3::Y * half_height),
                    transform.transform_point3(side + Vec3::Y * half_height),
                    COLLIDER_COLOR,
                );
            }
        }
        Collider::Plane { .. } => {}
    }
}

/// How big a light's marker is (m); also how near a click must come to
/// pick a light with no shape.
pub const LIGHT_MARKER: f32 = 0.2;

/// A light's marker in its own colour: a six-pointed star, plus its reach
/// when selected (rings for a point light, the cone for a spot).
fn push_light_marker(list: &mut DrawList, light: &Light, selected: bool) {
    let peak = light.color.max_element().max(1e-3);
    let color = (light.color / peak).extend(1.0).to_array();
    let p = light.position;
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        list.push_line(p - axis * LIGHT_MARKER, p + axis * LIGHT_MARKER, color);
    }
    if !selected {
        return;
    }
    let ring = |list: &mut DrawList, center: Vec3, radius: f32, u: Vec3, v: Vec3| {
        let point = |k: usize| {
            let (s, c) = (k as f32 / 32.0 * std::f32::consts::TAU).sin_cos();
            center + (u * c + v * s) * radius
        };
        for k in 0..32 {
            list.push_line(point(k), point(k + 1), color);
        }
    };
    match light.kind {
        LightKind::Point => {
            ring(list, p, light.range, Vec3::X, Vec3::Z);
            ring(list, p, light.range, Vec3::X, Vec3::Y);
            ring(list, p, light.range, Vec3::Y, Vec3::Z);
        }
        LightKind::Spot {
            direction,
            outer_angle,
            ..
        } => {
            let u = direction.any_orthonormal_vector();
            let v = direction.cross(u);
            let (s, c) = outer_angle.sin_cos();
            let center = p + direction * light.range * c;
            let radius = light.range * s;
            ring(list, center, radius, u, v);
            for k in 0..8 {
                let (ks, kc) = (k as f32 / 8.0 * std::f32::consts::TAU).sin_cos();
                list.push_line(p, center + (u * kc + v * ks) * radius, color);
            }
        }
    }
}

/// The 12 edges of a box of `half` extents placed by `transform`.
fn push_box_outline(list: &mut DrawList, transform: Mat4, half: Vec3) {
    push_box_outline_colored(list, transform, half, OUTLINE_COLOR);
}

fn push_box_outline_colored(list: &mut DrawList, transform: Mat4, half: Vec3, color: [f32; 4]) {
    let corner = |i: usize| {
        let s = |bit: usize| if i & bit == 0 { -1.0 } else { 1.0 };
        transform.transform_point3(half * Vec3::new(s(1), s(2), s(4)))
    };
    for i in 0..8 {
        for bit in [1, 2, 4] {
            if i & bit == 0 {
                list.push_line(corner(i), corner(i | bit), color);
            }
        }
    }
}

/// How to draw a frame of the scene.
#[derive(Debug, Clone, Copy)]
pub struct DrawOptions {
    /// The scene's surroundings (walls are drawn to match the pit).
    pub arena: Arena,
    /// Object to highlight (edit mode).
    pub selected: Option<usize>,
    /// Draw model objects' colliders as wireframes over their meshes.
    pub show_colliders: bool,
    /// Interpolation between the previous and current physics states.
    pub alpha: f32,
    pub show_grid: bool,
    pub show_axes: bool,
    /// Draw a marker at each light (and its reach, when selected).
    pub show_lights: bool,
}

pub fn build_draw_list(
    list: &mut DrawList,
    world: &World,
    objects: &[SceneObject],
    hull_meshes: &HullMeshes,
    model_meshes: &ModelMeshes,
    descendants: &[usize],
    options: DrawOptions,
) {
    let DrawOptions {
        arena,
        selected,
        show_colliders,
        alpha,
        show_grid,
        show_axes,
        show_lights,
    } = options;
    list.clear();
    push_scenery(list, arena);
    for (index, object) in objects.iter().enumerate() {
        let Some(body) = world.body(object.handle) else {
            continue;
        };
        let current = Pose {
            position: body.position(),
            orientation: body.orientation(),
        };
        let pose = interpolate(object.prev, current, alpha);
        let selected = selected == Some(index);
        if let Some(light) = &object.light {
            let light = light.at(pose.position, pose.orientation);
            list.push_light(light);
            if show_lights {
                push_light_marker(list, &light, selected);
            }
        }
        let Some(collider) = body.collider() else {
            continue;
        };
        let body_transform = Mat4::from_rotation_translation(pose.orientation, pose.position);
        let color = if selected {
            let [r, g, b, a] = object.color;
            let [r, g, b] = [r, g, b].map(|c| c + (1.0 - c) * SELECTED_TINT);
            [r, g, b, a]
        } else {
            object.color
        };
        match object.model {
            Some(visual) => {
                for &mesh in model_meshes.get(&visual.model).into_iter().flatten() {
                    // 1, 1: the model's own material decides.
                    list.push_instance_emissive(
                        mesh,
                        body_transform * visual.transform,
                        color,
                        1.0,
                        1.0,
                        object.emissive,
                    );
                }
                if show_colliders {
                    push_collider_wireframe(list, world, collider, body_transform);
                }
            }
            None => {
                for (mesh, local) in collider_visual(collider, hull_meshes) {
                    list.push_instance_emissive(
                        mesh,
                        body_transform * local,
                        color,
                        object.roughness,
                        object.metallic,
                        object.emissive,
                    );
                }
            }
        }
        if selected {
            push_box_outline(list, body_transform, local_half_extents(collider, world));
        } else if descendants.contains(&index) {
            // What moves with the selection.
            push_box_outline_colored(
                list,
                body_transform,
                local_half_extents(collider, world),
                DESCENDANT_OUTLINE,
            );
        }
        if show_axes {
            let length = collider_extent(collider);
            for (axis, color) in [
                (Vec3::X, [1.0, 0.2, 0.2, 1.0]),
                (Vec3::Y, [0.2, 1.0, 0.2, 1.0]),
                (Vec3::Z, [0.2, 0.4, 1.0, 1.0]),
            ] {
                let tip = pose.position + pose.orientation * axis * length;
                list.push_line(pose.position, tip, color);
            }
        }
    }
    // Joints whose anchors sit apart (distance joints) are drawn as ropes.
    for id in world.joints() {
        if let Some((a, b)) = world.joint_anchors(id)
            && a.distance(b) > 1e-3
        {
            list.push_line(a, b, ROPE_COLOR);
        }
    }
    if show_grid {
        let half = FLOOR_SIZE / 2.0;
        let y = 0.01; // just above the floor to avoid z-fighting
        for i in 0..=20 {
            let t = -half + i as f32 * 2.0;
            list.push_line(Vec3::new(t, y, -half), Vec3::new(t, y, half), GRID_COLOR);
            list.push_line(Vec3::new(-half, y, t), Vec3::new(half, y, t), GRID_COLOR);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCENERY: usize = 5; // floor + 4 walls

    #[test]
    fn pit_has_five_static_planes_and_bodies_start_inside() {
        let mut world = World::default();
        let objects = demo_scene(&mut world);
        assert_eq!(world.len(), 5 + BODY_COUNT);
        let statics = world
            .bodies()
            .filter(|(_, b)| b.inverse_mass() == 0.0)
            .count();
        assert_eq!(statics, 5);
        for o in &objects {
            let body = world.body(o.handle).unwrap();
            // Bounding radius: sphere radius, or the box's half-diagonal.
            let radius = match body.collider() {
                Some(Collider::Sphere { radius }) => radius,
                Some(Collider::Box { half_extents }) => half_extents.length(),
                Some(Collider::Capsule {
                    half_height,
                    radius,
                }) => half_height + radius,
                Some(Collider::Hull { hull }) => world
                    .hull(hull)
                    .unwrap()
                    .vertices()
                    .iter()
                    .map(|v| v.length())
                    .fold(0.0, f32::max),
                other => panic!("unexpected collider {other:?}"),
            };
            let p = body.position();
            assert!(
                p.x.abs() + radius < PIT && p.z.abs() + radius < PIT,
                "outside the pit: {p}"
            );
            assert!(p.y - radius > 0.0, "starts in the floor: {p}");
        }
    }

    #[test]
    fn hull_meshes_face_outward_and_the_pit_has_gems() {
        let mut world = World::default();
        build(&mut world, Demo::Pit);
        let hulls = hulls_in_world(&world);
        assert!(!hulls.is_empty(), "the pit should contain gems");
        for id in hulls {
            let mesh = hull_mesh(world.hull(id).unwrap());
            for tri in mesh.indices.chunks(3) {
                let [a, b, c] = [tri[0], tri[1], tri[2]]
                    .map(|i| Vec3::from(mesh.vertices[i as usize].position));
                let n = Vec3::from(mesh.vertices[tri[0] as usize].normal);
                assert!((b - a).cross(c - a).dot(n) > 0.0, "inward-wound triangle");
            }
        }
    }

    #[test]
    fn rain_drops_a_thousand_bodies_inside_its_pit() {
        let mut world = World::default();
        let objects = build(&mut world, Demo::Rain);
        assert_eq!(objects.len(), RAIN_COUNT);
        for o in &objects {
            let p = world.body(o.handle).unwrap().position();
            assert!(
                p.x.abs() < RAIN_PIT - 0.5 && p.z.abs() < RAIN_PIT - 0.5 && p.y > 0.5,
                "{p}"
            );
        }
        assert_eq!(Demo::from_name("RAIN"), Some(Demo::Rain));
        assert_eq!(Demo::from_name("nope"), None);
    }

    #[test]
    fn joints_scene_holds_together_and_stays_in_the_pit() {
        use crate::joint_scene::{CHAIN_LINKS, CRATES, PLANKS, RAGDOLL_PARTS};
        let mut world = World::default();
        let objects = build(&mut world, Demo::Joints);
        let posts = 3;
        let bridge_group = 1;
        assert_eq!(
            objects.len(),
            posts + bridge_group + CHAIN_LINKS + PLANKS + CRATES + RAGDOLL_PARTS
        );
        let gaps = |w: &World| -> Vec<f32> {
            w.joints()
                .map(|j| {
                    let (a, b) = w.joint_anchors(j).unwrap();
                    a.distance(b)
                })
                .collect()
        };
        let before = gaps(&world);
        for _ in 0..(6.0 * 60.0) as u32 {
            world.step(1.0 / 60.0);
        }
        for (start, end) in before.iter().zip(gaps(&world)) {
            assert!((end - start).abs() < 0.03, "joint gap {start} -> {end}");
        }
        for o in &objects {
            let p = world.body(o.handle).unwrap().position();
            assert!(
                p.is_finite() && p.y > 0.0 && p.x.abs() < PIT && p.z.abs() < PIT,
                "{p}"
            );
        }
        assert_eq!(Demo::from_name("joints"), Some(Demo::Joints));
    }

    #[test]
    fn every_demo_round_trips_through_ron_and_names_its_objects() {
        for demo in Demo::ALL {
            let desc = demo_desc(demo);
            let text = crate::scene_desc::to_ron(&desc);
            assert_eq!(
                crate::scene_desc::from_ron(&text).unwrap(),
                desc,
                "{demo:?}"
            );
            assert!(desc.objects.iter().all(|o| !o.name.is_empty()));
        }
    }

    #[test]
    fn materials_chart_spans_roughness_and_metallic() {
        let desc = demo_desc(Demo::Materials);
        let spheres: Vec<_> = desc
            .objects
            .iter()
            .filter(|o| matches!(o.shape, Some(ShapeDesc::Sphere { .. })))
            .collect();
        assert_eq!(spheres.len(), CHART_COLUMNS * CHART_ROWS);
        let range = |f: fn(&ObjectDesc) -> f32| {
            let values: Vec<f32> = spheres.iter().map(|o| f(o)).collect();
            (
                values.iter().copied().fold(f32::MAX, f32::min),
                values.iter().copied().fold(f32::MIN, f32::max),
            )
        };
        assert_eq!(range(|o| o.metallic), (0.0, 1.0));
        let (lo, hi) = range(|o| o.roughness);
        assert!(lo < 0.1 && hi == 1.0);
        let mut world = World::default();
        let objects = build(&mut world, Demo::Materials);
        assert_eq!(objects.len(), desc.objects.len());
        assert_eq!(Demo::from_name("materials"), Some(Demo::Materials));
    }

    #[test]
    fn props_scene_settles_inside_the_pit() {
        let mut world = World::default();
        let objects = build(&mut world, Demo::Props);
        assert_eq!(objects.len(), PROP_COUNT);
        assert!(objects.iter().all(|o| o.model.is_some()));
        for _ in 0..(6.0 * 60.0) as u32 {
            world.step(1.0 / 60.0);
        }
        for o in &objects {
            let p = world.body(o.handle).unwrap().position();
            assert!(
                p.is_finite() && p.y > 0.0 && p.x.abs() < PIT && p.z.abs() < PIT,
                "{p}"
            );
        }
        assert_eq!(Demo::from_name("props"), Some(Demo::Props));
    }

    #[test]
    fn the_empty_scene_has_a_finite_floor_things_can_fall_off() {
        let mut desc = empty_desc();
        let ball = |x: f32| ObjectDesc {
            shape: Some(ShapeDesc::Sphere { radius: 0.5 }),
            position: [x, 2.0, 0.0],
            ..ObjectDesc::default()
        };
        desc.objects.push(ball(0.0));
        desc.objects.push(ball(NEW_FLOOR_HALF + 3.0));
        let (mut world, objects) =
            crate::scene_desc::build_world(&desc, &crate::models::ModelRegistry::default())
                .unwrap();
        assert_eq!(
            world.len(),
            3,
            "the floor object and two balls; no arena planes"
        );
        for _ in 0..120 {
            world.step(1.0 / 60.0);
        }
        let y = |i: usize| world.body(objects[i].handle).unwrap().position().y;
        assert!((y(1) - 0.5).abs() < 0.02, "on the floor: {}", y(1));
        assert!(y(2) < -5.0, "past the edge it falls: {}", y(2));
        let desc = empty_desc();
        assert_eq!(
            crate::scene_desc::from_ron(&crate::scene_desc::to_ron(&desc)).unwrap(),
            desc
        );
    }

    /// Every demo written as a v1 file (world-space joints) loads, migrates,
    /// and builds the same world: bodies where authored, every joint's
    /// anchors at its authored world points.
    #[test]
    fn v1_demos_migrate_to_the_same_world() {
        use crate::scene_desc::{JointKindSpec, from_ron, to_ron};
        for demo in Demo::ALL {
            let v1 = demo_world_desc(demo);
            let migrated = from_ron(&to_ron(&v1)).unwrap();
            assert_eq!(migrated.version, SCENE_VERSION);
            let (world, objects) =
                crate::scene_desc::build_world(&migrated, &crate::models::ModelRegistry::bundled())
                    .unwrap();
            for (o, authored) in objects.iter().zip(&v1.objects) {
                let p = world.body(o.handle).unwrap().position();
                assert!(
                    p.abs_diff_eq(Vec3::from_array(authored.position), 1e-4),
                    "{demo:?}"
                );
            }
            for (id, joint) in world.joints().zip(&v1.joints) {
                let (pa, pb) = world.joint_anchors(id).unwrap();
                let (want_a, want_b) = match joint.kind {
                    JointKindSpec::BallSocket { pivot, .. }
                    | JointKindSpec::Hinge { pivot, .. } => {
                        (Vec3::from_array(pivot), Vec3::from_array(pivot))
                    }
                    JointKindSpec::Distance { anchor_a, anchor_b } => {
                        (Vec3::from_array(anchor_a), Vec3::from_array(anchor_b))
                    }
                };
                assert!(
                    pa.abs_diff_eq(want_a, 1e-4) && pb.abs_diff_eq(want_b, 1e-4),
                    "{demo:?}: {pa} {pb} vs {want_a} {want_b}"
                );
            }
        }
    }

    /// Grouping the Joints demo (ragdoll skeleton, bridge under an empty)
    /// changes nothing in the built world.
    #[test]
    fn the_grouped_joints_demo_builds_the_same_world_as_flat() {
        use crate::scene_desc::{build_world, from_ron, to_ron};
        let grouped = demo_desc(Demo::Joints);
        assert!(
            grouped
                .objects
                .iter()
                .filter(|o| o.parent.is_some())
                .count()
                >= 15
        );
        let flat = from_ron(&to_ron(&demo_world_desc(Demo::Joints))).unwrap();
        let registry = crate::models::ModelRegistry::default();
        let (a, oa) = build_world(&grouped, &registry).unwrap();
        let (b, ob) = build_world(&flat, &registry).unwrap();
        for (x, y) in oa.iter().zip(&ob) {
            let (x, y) = (a.body(x.handle).unwrap(), b.body(y.handle).unwrap());
            assert!(x.position().abs_diff_eq(y.position(), 1e-4));
            assert!(x.orientation().abs_diff_eq(y.orientation(), 1e-4));
        }
        for (i, j) in a.joints().zip(b.joints()) {
            let (pa, pb) = (a.joint_anchors(i).unwrap(), b.joint_anchors(j).unwrap());
            assert!(pa.0.abs_diff_eq(pb.0, 1e-4) && pa.1.abs_diff_eq(pb.1, 1e-4));
        }
    }

    #[test]
    fn stack_scenes_build_the_expected_bodies() {
        let mut world = World::default();
        assert_eq!(build(&mut world, Demo::Tower).len(), TOWER_HEIGHT);
        assert_eq!(world.len(), 5 + TOWER_HEIGHT);
        let pyramid = build(&mut world, Demo::Pyramid);
        assert_eq!(pyramid.len(), PYRAMID_BASE * (PYRAMID_BASE + 1) / 2);
        assert_eq!(
            world.len(),
            5 + pyramid.len(),
            "switching scenes replaces the old bodies"
        );
    }

    #[test]
    fn demo_scene_is_deterministic_and_replaces_old_bodies() {
        let mut world = World::default();
        let first = demo_scene(&mut world);
        let second = demo_scene(&mut world);
        assert_eq!(world.len(), 5 + BODY_COUNT);
        let poses = |objs: &[SceneObject]| objs.iter().map(|o| o.prev).collect::<Vec<_>>();
        assert_eq!(poses(&first), poses(&second));
        assert!(
            first.iter().all(|o| world.body(o.handle).is_none()),
            "old handles must be stale"
        );
    }

    #[test]
    fn lights_demo_hands_every_light_to_the_renderer_where_its_object_is() {
        let desc = demo_desc(Demo::Lights);
        assert_eq!(desc.sky, SkyPreset::Dusk);
        let (world, objects) =
            crate::scene_desc::build_world(&desc, &crate::models::ModelRegistry::default())
                .unwrap();
        let mut list = DrawList::default();
        let options = DrawOptions {
            arena: desc.arena,
            selected: None,
            show_colliders: false,
            alpha: 1.0,
            show_grid: false,
            show_axes: false,
            show_lights: false,
        };
        build_draw_list(
            &mut list,
            &world,
            &objects,
            &HullMeshes::new(),
            &ModelMeshes::new(),
            &[],
            options,
        );
        let carriers: Vec<&SceneObject> = objects.iter().filter(|o| o.light.is_some()).collect();
        assert_eq!(list.lights.len(), carriers.len());
        assert!(list.lights.len() >= 80, "{}", list.lights.len());
        for (light, object) in list.lights.iter().zip(&carriers) {
            assert_eq!(
                light.position,
                world.body(object.handle).unwrap().position()
            );
        }
        let spots = list
            .lights
            .iter()
            .filter(|l| matches!(l.kind, LightKind::Spot { .. }) && l.shadows)
            .count();
        assert_eq!(spots, 2);
        // Bulbs glow; markers only when asked for.
        assert!(list.instances.iter().any(|i| i.emissive > 0.0));
        let lines = list.lines.len();
        build_draw_list(
            &mut list,
            &world,
            &objects,
            &HullMeshes::new(),
            &ModelMeshes::new(),
            &[],
            DrawOptions {
                show_lights: true,
                ..options
            },
        );
        assert_eq!(list.lines.len(), lines + 3 * carriers.len());
    }

    #[test]
    fn scatters_are_deterministic_sized_and_clear_of_the_middle() {
        let models = ModelRegistry::bundled();
        // Mesh ids stand in for uploaded meshes.
        let meshes: ModelMeshes = models
            .list()
            .map(|(id, _, _)| (id, vec![MeshId(100 + id.0)]))
            .collect();
        let scatter = Scatter {
            count: 2_000,
            half_extent: 40.0,
            seed: 7,
        };
        let a = scatter_instances(&scatter, &models, &meshes);
        assert_eq!(a, scatter_instances(&scatter, &models, &meshes));
        assert_ne!(
            a,
            scatter_instances(&Scatter { seed: 8, ..scatter }, &models, &meshes)
        );
        // Every bundled model here has one mesh: one instance per prop.
        assert_eq!(a.len(), 2_000);
        let modelled = a.iter().filter(|i| i.mesh.0 >= 100).count();
        assert!((1_200..1_600).contains(&modelled), "{modelled} model props");
        for i in &a {
            let p = i.transform.w_axis.truncate();
            assert!(p.x.abs() <= 40.0 && p.z.abs() <= 40.0, "{p}");
            assert!(p.x * p.x + p.z * p.z >= 36.0, "{p} in the middle");
            assert!((0.0..2.0).contains(&p.y), "{p} off the ground");
        }
        // Without models, plain shapes only; and a tiny field still fills.
        let plain = scatter_instances(&scatter, &ModelRegistry::default(), &ModelMeshes::new());
        assert!(plain.iter().all(|i| i.mesh.0 < MeshKind::ALL.len() as u32));
        let tiny = Scatter {
            count: 50,
            half_extent: 1.0,
            seed: 0,
        };
        assert_eq!(scatter_instances(&tiny, &models, &meshes).len(), 50);
    }

    #[test]
    fn draw_list_has_scenery_and_every_ball_at_the_interpolated_pose() {
        let mut world = World::default();
        let mut objects = demo_scene(&mut world);
        snapshot(&world, &mut objects);
        world.step(0.1);
        let mut list = DrawList::default();
        build_draw_list(
            &mut list,
            &world,
            &objects,
            &HullMeshes::new(),
            &ModelMeshes::new(),
            &[],
            DrawOptions {
                arena: Arena::Pit { half: PIT },
                selected: None,
                show_colliders: false,
                alpha: 0.0,
                show_grid: true,
                show_axes: false,
                show_lights: false,
            },
        );
        // One instance per body, except capsules: cylinder + two end spheres.
        let expected: usize = objects
            .iter()
            .map(|o| {
                collider_visual(
                    world.body(o.handle).unwrap().collider().unwrap(),
                    &HullMeshes::new(),
                )
                .len()
            })
            .sum();
        assert!(expected > BODY_COUNT, "the pit should include capsules");
        assert_eq!(list.instances.len(), SCENERY + expected);
        assert_eq!(list.instances[0].mesh, MeshId::from(MeshKind::Plane));
        // alpha 0 draws the pre-step pose, not the current one.
        assert_eq!(
            list.instances[SCENERY].transform.w_axis.y,
            objects[0].prev.position.y
        );
        assert_eq!(list.lines.len(), 42);
    }

    #[test]
    fn bodies_settle_inside_the_pit() {
        let mut world = World::default();
        let objects = demo_scene(&mut world);
        for _ in 0..(8 * 60) {
            world.step(1.0 / 60.0);
        }
        assert!(!all_below(&world, &objects, -30.0));
        for o in &objects {
            let p = world.body(o.handle).unwrap().position();
            assert!(
                p.is_finite() && p.y > -0.1 && p.x.abs() < PIT + 0.1 && p.z.abs() < PIT + 0.1,
                "{p}"
            );
        }
    }
}
