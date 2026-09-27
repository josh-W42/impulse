//! Editor UI and viewport helpers: the inspector window, new objects, picking,
//! and dragging on a plane. Geometry helpers are pure so they can be tested.

use glam::Vec3;
use impulse_physics::World;

use std::collections::HashSet;

use crate::editor::{Editor, ReparentError, is_within};
use crate::scene::{SceneObject, default_name};
use crate::scene_desc::{BodyType, JointKindSpec, ModelCollider, ObjectDesc, SceneDesc, ShapeDesc};

/// Requests from the inspector that the app carries out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InspectorAction {
    Duplicate,
    Delete,
    /// Move the selection under this parent (`None`: top level).
    Reparent(Option<usize>),
}

/// Shapes the editor can add.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewShape {
    Sphere,
    Box,
    Capsule,
}

impl NewShape {
    pub const ALL: [NewShape; 3] = [NewShape::Sphere, NewShape::Box, NewShape::Capsule];

    pub fn label(self) -> &'static str {
        match self {
            NewShape::Sphere => "Sphere",
            NewShape::Box => "Box",
            NewShape::Capsule => "Capsule",
        }
    }
}

const NEW_COLORS: [[f32; 3]; 4] = [
    [0.85, 0.45, 0.30],
    [0.35, 0.65, 0.45],
    [0.35, 0.55, 0.85],
    [0.85, 0.75, 0.35],
];

/// A new dynamic object resting `at` (its center), named after its index.
pub fn new_object(shape: NewShape, at: Vec3, index: usize) -> ObjectDesc {
    let shape = match shape {
        NewShape::Sphere => ShapeDesc::Sphere { radius: 0.5 },
        NewShape::Box => ShapeDesc::Box {
            half_extents: [0.5; 3],
        },
        NewShape::Capsule => ShapeDesc::Capsule {
            half_height: 0.4,
            radius: 0.3,
        },
    };
    let mut object = ObjectDesc {
        shape: Some(shape),
        position: at.to_array(),
        color: NEW_COLORS[index % NEW_COLORS.len()],
        ..ObjectDesc::default()
    };
    object.name = default_name(&object, index);
    object
}

/// A new dynamic model object scaled to about 1 m, resting with its bottom at
/// `floor` (white, so the texture shows its own colours).
pub fn new_model_object(
    registry: &crate::models::ModelRegistry,
    id: crate::models::ModelId,
    at: Vec3,
    floor: f32,
    index: usize,
) -> Option<ObjectDesc> {
    let asset = registry.get(id)?;
    let (_, name, _) = registry.list().find(|(m, _, _)| *m == id)?;
    let scale = crate::models::fitting_scale(asset, 1.0);
    let half_height = (asset.data.max.y - asset.data.min.y) / 2.0 * scale;
    let mut object = ObjectDesc {
        shape: Some(ShapeDesc::Model {
            name: name.to_string(),
            scale,
            collider: ModelCollider::Hull,
        }),
        position: [at.x, floor + half_height + 0.05, at.z],
        color: [1.0; 3],
        ..ObjectDesc::default()
    };
    object.name = default_name(&object, index);
    Some(object)
}

/// The object whose body the ray hits first (arena planes don't count).
pub fn object_at(
    world: &World,
    objects: &[SceneObject],
    origin: Vec3,
    dir: Vec3,
) -> Option<(usize, Vec3)> {
    let hit = world.raycast(origin, dir, 500.0)?;
    let index = objects.iter().position(|o| o.handle == hit.body)?;
    Some((index, hit.point))
}

/// Where the ray meets the plane through `point` with `normal`; `None` if it
/// runs parallel or the plane is behind the ray.
pub fn ray_plane(origin: Vec3, dir: Vec3, point: Vec3, normal: Vec3) -> Option<Vec3> {
    let denom = dir.dot(normal);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (point - origin).dot(normal) / denom;
    (t >= 0.0).then(|| origin + dir * t)
}

/// Shape choices in the inspector (a gem can be replaced but not created).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShapeKind {
    Anchor,
    Sphere,
    Box,
    Capsule,
    Gem,
    Model,
}

impl ShapeKind {
    fn of(shape: &Option<ShapeDesc>) -> Self {
        match shape {
            None => Self::Anchor,
            Some(ShapeDesc::Sphere { .. }) => Self::Sphere,
            Some(ShapeDesc::Box { .. }) => Self::Box,
            Some(ShapeDesc::Capsule { .. }) => Self::Capsule,
            Some(ShapeDesc::Hull { .. }) => Self::Gem,
            Some(ShapeDesc::Model { .. }) => Self::Model,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Anchor => "none (empty)",
            Self::Sphere => "sphere",
            Self::Box => "box",
            Self::Capsule => "capsule",
            Self::Gem => "gem (hull)",
            Self::Model => "model",
        }
    }

    /// The new shape, sized like the old one.
    fn replacing(self, old: &Option<ShapeDesc>) -> Option<ShapeDesc> {
        let size = match old {
            None => 0.5,
            Some(ShapeDesc::Sphere { radius }) => *radius,
            Some(ShapeDesc::Box { half_extents }) => {
                half_extents.iter().copied().fold(0.0, f32::max)
            }
            Some(ShapeDesc::Capsule {
                half_height,
                radius,
            }) => half_height + radius,
            Some(ShapeDesc::Hull { points }) => points
                .iter()
                .map(|p| Vec3::from_array(*p).length())
                .fold(0.0, f32::max),
            // The model's size isn't known here; start from a unit-ish shape.
            Some(ShapeDesc::Model { scale, .. }) => 0.5 * scale,
        }
        .max(0.05);
        match self {
            Self::Anchor => None,
            Self::Sphere => Some(ShapeDesc::Sphere { radius: size }),
            Self::Box => Some(ShapeDesc::Box {
                half_extents: [size * 0.8; 3],
            }),
            Self::Capsule => Some(ShapeDesc::Capsule {
                half_height: size * 0.6,
                radius: size * 0.4,
            }),
            Self::Gem | Self::Model => old.clone(),
        }
    }
}

fn number(ui: &mut egui::Ui, value: &mut f32, speed: f32, range: std::ops::RangeInclusive<f32>) {
    ui.add(egui::DragValue::new(value).speed(speed).range(range));
}

fn vec3_row(ui: &mut egui::Ui, label: &str, v: &mut [f32; 3], speed: f32, min: f32) {
    ui.horizontal(|ui| {
        ui.label(label);
        for c in v.iter_mut() {
            number(ui, c, speed, min..=1000.0);
        }
    });
}

fn joint_label(kind: &JointKindSpec) -> &'static str {
    match kind {
        JointKindSpec::BallSocket { .. } => "ball-socket",
        JointKindSpec::Hinge { .. } => "hinge",
        JointKindSpec::Distance { .. } => "distance",
    }
}

/// Indices of the objects whose names contain `filter` (ignoring case), in
/// scene order; every object when the filter is blank.
pub fn filtered_objects(desc: &SceneDesc, filter: &str) -> Vec<usize> {
    let needle = filter.trim().to_lowercase();
    desc.objects
        .iter()
        .enumerate()
        .filter(|(_, o)| needle.is_empty() || o.name.to_lowercase().contains(&needle))
        .map(|(i, _)| i)
        .collect()
}

fn kind_tag(shape: &Option<ShapeDesc>) -> &'static str {
    match ShapeKind::of(shape) {
        ShapeKind::Anchor => "empty",
        ShapeKind::Sphere => "sphere",
        ShapeKind::Box => "box",
        ShapeKind::Capsule => "capsule",
        ShapeKind::Gem => "gem",
        ShapeKind::Model => "model",
    }
}

/// One row of the objects tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeRow {
    pub index: usize,
    pub depth: usize,
    pub has_children: bool,
}

/// The objects in tree order: each object followed by its children (list
/// order), skipping the children of `collapsed` objects. Objects unreachable
/// from the top level (only possible with a cycle) are listed at depth 0.
pub fn tree_rows(desc: &SceneDesc, collapsed: &HashSet<usize>) -> Vec<TreeRow> {
    let n = desc.objects.len();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut roots = Vec::new();
    for (i, object) in desc.objects.iter().enumerate() {
        match object.parent {
            Some(p) if p < n && p != i => children[p].push(i),
            _ => roots.push(i),
        }
    }
    let mut rows = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    let mut stack: Vec<(usize, usize)> = roots.iter().rev().map(|&r| (r, 0)).collect();
    loop {
        while let Some((i, depth)) = stack.pop() {
            if std::mem::replace(&mut seen[i], true) {
                continue;
            }
            rows.push(TreeRow {
                index: i,
                depth,
                has_children: !children[i].is_empty(),
            });
            if !collapsed.contains(&i) {
                stack.extend(children[i].iter().rev().map(|&c| (c, depth + 1)));
            }
        }
        // Anything left is in a cycle (or under a collapsed row): only
        // cycles need listing.
        match (0..n).find(|&i| !seen[i] && !under_collapsed(desc, i, collapsed)) {
            Some(i) => stack.push((i, 0)),
            None => break,
        }
    }
    rows
}

fn under_collapsed(desc: &SceneDesc, i: usize, collapsed: &HashSet<usize>) -> bool {
    let mut current = desc.objects[i].parent;
    for _ in 0..desc.objects.len() {
        match current {
            Some(p) if p < desc.objects.len() => {
                if collapsed.contains(&p) {
                    return true;
                }
                current = desc.objects[p].parent;
            }
            _ => return false,
        }
    }
    false
}

/// "bridge › plank 4": an object's ancestors and name.
pub fn path(desc: &SceneDesc, index: usize) -> String {
    let mut names = vec![desc.objects[index].name.as_str()];
    let mut current = desc.objects[index].parent;
    for _ in 0..desc.objects.len() {
        let Some(p) = current.filter(|&p| p < desc.objects.len()) else {
            break;
        };
        names.push(desc.objects[p].name.as_str());
        current = desc.objects[p].parent;
    }
    names.reverse();
    names.join(" \u{203a} ")
}

/// Payload of a row being dragged (the object index).
#[derive(Debug, Clone, Copy)]
struct RowDrag(usize);

/// The objects tree: click a row to select; drag a row onto another to make
/// it that object's child, or onto "(top level)" to unparent (the object
/// stays where it is in the world). With a filter, matches are listed flat
/// with their paths. `follow` scrolls to (and expands down to) the selection.
/// Returns a message when a drop is refused.
pub fn outliner(
    ctx: &egui::Context,
    editor: &mut Editor,
    filter: &mut String,
    collapsed: &mut HashSet<usize>,
    follow: bool,
) -> Option<String> {
    let mut message = None;
    egui::Window::new("objects")
        .anchor(egui::Align2::LEFT_BOTTOM, [12.0, -12.0])
        .resizable(false)
        .show(ctx, |ui| {
            ui.add(
                egui::TextEdit::singleline(filter)
                    .hint_text("filter by name")
                    .desired_width(200.0),
            );
            let filtering = !filter.trim().is_empty();
            if follow && let Some(selected) = editor.selected() {
                // Show the selection: expand its ancestors.
                // Bounded, so a (corrupt) parent loop can't hang it.
                let mut current = editor.desc().objects.get(selected).and_then(|o| o.parent);
                for _ in 0..editor.desc().objects.len() {
                    let Some(p) = current else { break };
                    collapsed.remove(&p);
                    current = editor.desc().objects.get(p).and_then(|o| o.parent);
                }
            }
            let rows: Vec<TreeRow> = if filtering {
                filtered_objects(editor.desc(), filter)
                    .into_iter()
                    .map(|index| TreeRow {
                        index,
                        depth: 0,
                        has_children: false,
                    })
                    .collect()
            } else {
                tree_rows(editor.desc(), collapsed)
            };
            let row_height =
                ui.text_style_height(&egui::TextStyle::Button) + ui.spacing().item_spacing.y;
            let mut area = egui::ScrollArea::vertical()
                .max_height(ROWS_SHOWN * row_height)
                .auto_shrink([false, true]);
            if follow
                && let Some(selected) = editor.selected()
                && let Some(row) = rows.iter().position(|r| r.index == selected)
            {
                area = area.vertical_scroll_offset((row as f32 - 2.0).max(0.0) * row_height);
            }
            let mut clicked = None;
            let mut toggled = None;
            let mut dropped: Option<(usize, Option<usize>)> = None;
            if !filtering {
                let top = ui.weak("(top level) — drop here to unparent");
                let top = ui.interact(top.rect, top.id.with("drop"), egui::Sense::hover());
                if let Some(drag) = top.dnd_release_payload::<RowDrag>() {
                    dropped = Some((drag.0, None));
                } else if top.dnd_hover_payload::<RowDrag>().is_some() {
                    ui.painter().rect_stroke(
                        top.rect,
                        2.0,
                        ui.visuals().selection.stroke,
                        egui::StrokeKind::Outside,
                    );
                }
            }
            area.show_rows(ui, row_height, rows.len(), |ui, range| {
                for row in &rows[range] {
                    let object = &editor.desc().objects[row.index];
                    let label = if filtering {
                        format!(
                            "{}   {}",
                            path(editor.desc(), row.index),
                            kind_tag(&object.shape)
                        )
                    } else {
                        format!("{}   {}", object.name, kind_tag(&object.shape))
                    };
                    let selected = editor.selected() == Some(row.index);
                    ui.horizontal(|ui| {
                        ui.add_space(row.depth as f32 * INDENT);
                        if row.has_children {
                            // Plain +/−: egui's default font has no ▸/▾.
                            let arrow = if collapsed.contains(&row.index) {
                                "+"
                            } else {
                                "\u{2212}"
                            };
                            if ui.small_button(arrow).clicked() {
                                toggled = Some(row.index);
                            }
                        } else if !filtering {
                            ui.add_space(ARROW_WIDTH);
                        }
                        let response = ui
                            .selectable_label(selected, label)
                            .interact(egui::Sense::drag());
                        response.dnd_set_drag_payload(RowDrag(row.index));
                        if response.clicked() {
                            clicked = Some(row.index);
                        }
                        if let Some(drag) = response.dnd_release_payload::<RowDrag>() {
                            dropped = Some((drag.0, Some(row.index)));
                        } else if response.dnd_hover_payload::<RowDrag>().is_some() {
                            ui.painter().rect_stroke(
                                response.rect,
                                2.0,
                                ui.visuals().selection.stroke,
                                egui::StrokeKind::Outside,
                            );
                        }
                    });
                }
            });
            if rows.is_empty() {
                ui.label("no matching objects");
            }
            if let Some(index) = toggled
                && !collapsed.remove(&index)
            {
                collapsed.insert(index);
            }
            if let Some(index) = clicked {
                editor.select(Some(index));
            }
            if let Some((child, parent)) = dropped
                && Some(child) != parent
            {
                match editor.reparent(child, parent) {
                    Ok(_) => editor.select(Some(child)),
                    Err(ReparentError::IntoOwnSubtree) => {
                        message = Some(format!(
                            "can't put '{}' inside itself or its own children",
                            editor.desc().objects[child].name
                        ));
                    }
                    Err(ReparentError::NoSuchObject) => {}
                }
            }
        });
    message
}

/// Indentation per tree level, and the width of an expand arrow (points).
const INDENT: f32 = 14.0;
const ARROW_WIDTH: f32 = 18.0;

/// Rows visible in the objects list before it scrolls.
const ROWS_SHOWN: f32 = 12.0;

/// The inspector for the selected object; edits go straight into `editor` as
/// part of the current gesture (the caller commits when interaction ends).
pub fn inspector(
    ctx: &egui::Context,
    editor: &mut Editor,
    model_names: &[&str],
) -> Option<InspectorAction> {
    let index = editor.selected()?;
    let original = editor.desc().objects[index].clone();
    let mut o = original.clone();
    let joints: Vec<(usize, String)> = editor
        .desc()
        .joints
        .iter()
        .enumerate()
        .filter(|(_, j)| j.a == index || j.b == index)
        .map(|(k, j)| {
            let other = if j.a == index { j.b } else { j.a };
            let name = editor
                .desc()
                .objects
                .get(other)
                .map_or("?", |o| o.name.as_str());
            (k, format!("{} to '{name}'", joint_label(&j.kind)))
        })
        .collect();
    let mut action = None;
    let mut remove_joint = None;
    let editor_names: Vec<String> = editor
        .desc()
        .objects
        .iter()
        .map(|o| o.name.clone())
        .collect();
    let candidate_parent: Vec<bool> = (0..editor_names.len())
        .map(|p| !is_within(editor.desc(), p, index))
        .collect();
    egui::Window::new("inspector")
        .anchor(egui::Align2::RIGHT_TOP, [-12.0, 12.0])
        .resizable(false)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("name");
                ui.text_edit_singleline(&mut o.name);
            });
            let current = ShapeKind::of(&o.shape);
            let mut kind = current;
            egui::ComboBox::from_label("shape")
                .selected_text(kind.label())
                .show_ui(ui, |ui| {
                    let mut choices = vec![
                        ShapeKind::Sphere,
                        ShapeKind::Box,
                        ShapeKind::Capsule,
                        ShapeKind::Anchor,
                    ];
                    if matches!(current, ShapeKind::Gem | ShapeKind::Model) {
                        choices.push(current);
                    }
                    for choice in choices {
                        ui.selectable_value(&mut kind, choice, choice.label());
                    }
                });
            if kind != current {
                o.shape = kind.replacing(&o.shape);
            }
            match &mut o.shape {
                Some(ShapeDesc::Sphere { radius }) => {
                    ui.horizontal(|ui| {
                        ui.label("radius");
                        number(ui, radius, 0.01, 0.01..=50.0);
                    });
                }
                Some(ShapeDesc::Box { half_extents }) => {
                    vec3_row(ui, "half size", half_extents, 0.01, 0.01);
                }
                Some(ShapeDesc::Capsule {
                    half_height,
                    radius,
                }) => {
                    ui.horizontal(|ui| {
                        ui.label("half height");
                        number(ui, half_height, 0.01, 0.01..=50.0);
                        ui.label("radius");
                        number(ui, radius, 0.01, 0.01..=50.0);
                    });
                }
                Some(ShapeDesc::Hull { points }) => {
                    ui.label(format!("{} points", points.len()));
                }
                Some(ShapeDesc::Model {
                    name,
                    scale,
                    collider,
                }) => {
                    egui::ComboBox::from_label("model")
                        .selected_text(name.as_str())
                        .show_ui(ui, |ui| {
                            for &choice in model_names {
                                if ui.selectable_label(name == choice, choice).clicked() {
                                    *name = choice.to_string();
                                }
                            }
                        });
                    ui.horizontal(|ui| {
                        ui.label("scale");
                        number(ui, scale, 0.01, 0.01..=100.0);
                        egui::ComboBox::from_label("collider")
                            .selected_text(format!("{collider:?}"))
                            .show_ui(ui, |ui| {
                                for kind in [
                                    ModelCollider::Hull,
                                    ModelCollider::Box,
                                    ModelCollider::Sphere,
                                    ModelCollider::Capsule,
                                ] {
                                    ui.selectable_value(collider, kind, format!("{kind:?}"));
                                }
                            });
                    });
                }
                None => {
                    ui.label("empty: invisible, no collisions (a folder or joint anchor)");
                }
            }
            // Parent: anything but this object and its descendants.
            let parent_name = |p: Option<usize>| match p {
                Some(p) => editor_names.get(p).cloned().unwrap_or_default(),
                None => "(top level)".to_string(),
            };
            egui::ComboBox::from_label("parent")
                .selected_text(parent_name(o.parent))
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(o.parent.is_none(), "(top level)")
                        .clicked()
                        && o.parent.is_some()
                    {
                        action = Some(InspectorAction::Reparent(None));
                    }
                    for (p, name) in editor_names.iter().enumerate() {
                        if candidate_parent[p]
                            && ui.selectable_label(o.parent == Some(p), name).clicked()
                            && o.parent != Some(p)
                        {
                            action = Some(InspectorAction::Reparent(Some(p)));
                        }
                    }
                });
            let relative = match o.parent {
                Some(p) => format!(" (relative to '{}')", parent_name(Some(p))),
                None => String::new(),
            };
            vec3_row(
                ui,
                &format!("position{relative}"),
                &mut o.position,
                0.05,
                -1000.0,
            );
            vec3_row(ui, "rotation°", &mut o.rotation_deg, 1.0, -1000.0);
            let mut fixed = matches!(o.body, BodyType::Static);
            if ui.checkbox(&mut fixed, "static").changed() {
                o.body = if fixed {
                    BodyType::Static
                } else {
                    BodyType::Dynamic { mass: 1.0 }
                };
            }
            if let BodyType::Dynamic { mass } = &mut o.body {
                ui.horizontal(|ui| {
                    ui.label("mass");
                    number(ui, mass, 0.05, 0.01..=1000.0);
                });
            }
            ui.horizontal(|ui| {
                ui.label("bounce");
                number(ui, &mut o.restitution, 0.01, 0.0..=1.5);
                ui.label("friction");
                number(ui, &mut o.friction, 0.01, 0.0..=2.0);
                ui.label("color");
                // The picker round-trips through other color types each frame,
                // nudging the low bits; only take the value on a real change.
                let mut color = o.color;
                if ui.color_edit_button_rgb(&mut color).changed() {
                    o.color = color;
                }
            });
            if !matches!(o.shape, Some(ShapeDesc::Model { .. }) | None) {
                ui.horizontal(|ui| {
                    ui.label("roughness");
                    ui.add(egui::Slider::new(&mut o.roughness, 0.0..=1.0));
                });
                ui.horizontal(|ui| {
                    ui.label("metallic");
                    ui.add(egui::Slider::new(&mut o.metallic, 0.0..=1.0));
                });
            }
            vec3_row(ui, "velocity", &mut o.velocity, 0.05, -1000.0);
            vec3_row(ui, "spin", &mut o.angular_velocity, 0.05, -1000.0);
            if !joints.is_empty() {
                ui.separator();
                for (k, label) in &joints {
                    ui.horizontal(|ui| {
                        ui.label(label);
                        if ui.small_button("remove").clicked() {
                            remove_joint = Some(*k);
                        }
                    });
                }
            }
            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Duplicate (Ctrl+D)").clicked() {
                    action = Some(InspectorAction::Duplicate);
                }
                if ui.button("Delete (Del)").clicked() {
                    action = Some(InspectorAction::Delete);
                }
            });
        });
    if o != original {
        editor.modify(|d| d.objects[index] = o);
    }
    if let Some(k) = remove_joint {
        editor.edit(|d| {
            d.joints.remove(k);
        });
    }
    action
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_desc::{Arena, SCENE_VERSION, SceneDesc, build_world};
    use std::collections::HashSet;

    #[test]
    fn picking_finds_the_object_under_the_ray_and_ignores_the_arena() {
        let desc = SceneDesc {
            version: SCENE_VERSION,
            name: "pick".into(),
            arena: Arena::Pit { half: 6.0 },
            objects: vec![
                new_object(NewShape::Box, Vec3::new(-2.0, 1.0, 0.0), 0),
                new_object(NewShape::Sphere, Vec3::new(2.0, 1.0, 0.0), 1),
            ],
            joints: Vec::new(),
        };
        let (world, objects) =
            build_world(&desc, &crate::models::ModelRegistry::default()).unwrap();
        let eye = Vec3::new(0.0, 1.0, 10.0);
        let toward = |p: Vec3| (p - eye).normalize();
        let (index, point) =
            object_at(&world, &objects, eye, toward(Vec3::new(2.0, 1.0, 0.0))).unwrap();
        assert_eq!(index, 1);
        assert!((point.distance(Vec3::new(2.0, 1.0, 0.0)) - 0.5).abs() < 1e-3);
        assert_eq!(
            object_at(&world, &objects, eye, toward(Vec3::new(-2.0, 1.0, 0.0))).map(|h| h.0),
            Some(0)
        );
        assert_eq!(object_at(&world, &objects, eye, Vec3::NEG_Y), None, "floor");
    }

    /// Idle frames must not touch the scene (each touch rebuilds the world).
    #[test]
    fn the_inspector_changes_nothing_by_itself() {
        let ctx = egui::Context::default();
        for demo in crate::scene::Demo::ALL {
            let mut editor = Editor::new(crate::scene::demo_desc(demo));
            for index in 0..editor.desc().objects.len().min(12) {
                editor.select(Some(index));
                let revision = editor.revision();
                for _ in 0..3 {
                    let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
                        inspector(ui.ctx(), &mut editor, &["barrel", "box"]);
                    });
                    output.textures_delta.clear(); // no renderer to apply them
                }
                assert_eq!(editor.revision(), revision, "{demo:?} object {index}");
            }
        }
    }

    #[test]
    fn the_filter_matches_names_ignoring_case_and_tracks_edits() {
        let mut editor = Editor::new(crate::scene::demo_desc(crate::scene::Demo::Joints));
        let all = filtered_objects(editor.desc(), "  ");
        assert_eq!(all.len(), editor.desc().objects.len());
        let boxes = filtered_objects(editor.desc(), "BOX");
        assert!(!boxes.is_empty());
        assert!(
            boxes
                .iter()
                .all(|&i| editor.desc().objects[i].name.contains("box"))
        );
        assert!(filtered_objects(editor.desc(), "no such thing").is_empty());

        // Rows always index the current scene: after a delete and an undo the
        // same row selects the same object again.
        let first_box = boxes[0];
        let name = editor.desc().objects[first_box].name.clone();
        editor.select(Some(first_box));
        editor.delete_selected();
        assert!(!filtered_objects(editor.desc(), &name).contains(&first_box));
        editor.undo();
        let row = filtered_objects(editor.desc(), &name);
        assert_eq!(row, [first_box]);
        editor.select(Some(row[0]));
        assert_eq!(editor.desc().objects[editor.selected().unwrap()].name, name);
    }

    /// The list's idle frames must not touch the scene either.
    #[test]
    fn the_outliner_changes_nothing_by_itself() {
        let ctx = egui::Context::default();
        let mut editor = Editor::new(crate::scene::demo_desc(crate::scene::Demo::Rain));
        editor.select(Some(500));
        let mut filter = String::new();
        for follow in [true, false, false] {
            let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
                outliner(
                    ui.ctx(),
                    &mut editor,
                    &mut filter,
                    &mut HashSet::new(),
                    follow,
                );
            });
            output.textures_delta.clear();
        }
        assert_eq!(editor.revision(), 0);
        assert_eq!(editor.selected(), Some(500));
    }

    #[test]
    fn new_models_are_about_a_metre_and_rest_on_the_floor() {
        let registry = crate::models::ModelRegistry::bundled();
        let id = registry.id("barrel").unwrap();
        let object = new_model_object(&registry, id, Vec3::new(2.0, 9.0, 3.0), 1.0, 4).unwrap();
        let Some(ShapeDesc::Model {
            name,
            scale,
            collider,
        }) = &object.shape
        else {
            panic!("expected a model");
        };
        assert_eq!((name.as_str(), *collider), ("barrel", ModelCollider::Hull));
        let size =
            (registry.get(id).unwrap().data.max - registry.get(id).unwrap().data.min) * *scale;
        assert!((size.max_element() - 1.0).abs() < 0.01, "{size}");
        assert!((object.position[1] - (1.0 + size.y / 2.0 + 0.05)).abs() < 1e-4);
        assert_eq!(object.name, "barrel 4");
        assert_eq!(object.color, [1.0; 3]);
    }

    fn tree_desc() -> SceneDesc {
        let node = |name: &str, parent: Option<usize>| ObjectDesc {
            name: name.into(),
            parent,
            ..ObjectDesc::default()
        };
        SceneDesc {
            version: SCENE_VERSION,
            name: "t".into(),
            arena: Arena::Empty,
            // Children listed before their parents too: tree order comes from
            // the links, not the list.
            objects: vec![
                node("b", Some(2)),
                node("x", None),
                node("a", None),
                node("c", Some(0)),
                node("d", Some(2)),
            ],
            joints: Vec::new(),
        }
    }

    #[test]
    fn tree_rows_nest_children_under_parents() {
        let desc = tree_desc();
        let rows = tree_rows(&desc, &HashSet::new());
        let shape: Vec<_> = rows
            .iter()
            .map(|r| (desc.objects[r.index].name.as_str(), r.depth))
            .collect();
        assert_eq!(shape, [("x", 0), ("a", 0), ("b", 1), ("c", 2), ("d", 1)]);
        assert!(rows[1].has_children && !rows[4].has_children);

        let collapsed: HashSet<usize> = [2].into();
        let names: Vec<_> = tree_rows(&desc, &collapsed)
            .iter()
            .map(|r| desc.objects[r.index].name.as_str())
            .collect();
        assert_eq!(names, ["x", "a"], "a collapsed hides b, c, d");
        assert_eq!(path(&desc, 3), "a \u{203a} b \u{203a} c");
    }

    #[test]
    fn tree_rows_survive_a_cycle() {
        let mut desc = tree_desc();
        desc.objects[1].parent = Some(4); // x under d
        desc.objects[2].parent = Some(1); // a under x: a -> x -> d -> a
        let rows = tree_rows(&desc, &HashSet::new());
        assert_eq!(rows.len(), desc.objects.len(), "every object listed once");
    }

    #[test]
    fn ray_plane_meets_ahead_only() {
        let hit = ray_plane(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0).normalize(),
            Vec3::ZERO,
            Vec3::Y,
        );
        assert!(hit.unwrap().abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-4));
        assert_eq!(
            ray_plane(Vec3::Y, Vec3::Y, Vec3::ZERO, Vec3::Y),
            None,
            "behind"
        );
        assert_eq!(
            ray_plane(Vec3::Y, Vec3::X, Vec3::ZERO, Vec3::Y),
            None,
            "parallel"
        );
    }

    #[test]
    fn shape_changes_keep_the_rough_size() {
        let sphere = Some(ShapeDesc::Sphere { radius: 2.0 });
        assert_eq!(
            ShapeKind::Box.replacing(&sphere),
            Some(ShapeDesc::Box {
                half_extents: [1.6; 3]
            })
        );
        assert_eq!(ShapeKind::Anchor.replacing(&sphere), None);
        assert_eq!(
            ShapeKind::Sphere.replacing(&None),
            Some(ShapeDesc::Sphere { radius: 0.5 })
        );
    }
}
