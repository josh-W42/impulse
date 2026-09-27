//! Editor state with no UI: the authored scene, the selection, and undo/redo.
//! Undo stores whole-description snapshots (scenes are small data).
//!
//! Continuous edits (dragging a value or a body) are one undo step: `begin`,
//! any number of `modify`, then `commit`. `edit` does all three at once.

use glam::{Quat, Vec3};

use crate::scene_desc::{
    BodyType, JointSpec, ObjectDesc, SceneDesc, set_parent_keep_world, world_poses,
};

pub const UNDO_DEPTH: usize = 100;

/// How far a duplicate is placed from its original (m).
const DUPLICATE_OFFSET: Vec3 = Vec3::new(0.5, 0.0, 0.5);

pub struct Editor {
    desc: SceneDesc,
    selected: Option<usize>,
    undo: Vec<SceneDesc>,
    redo: Vec<SceneDesc>,
    /// The description when the current gesture began.
    pending: Option<SceneDesc>,
    /// Bumped on every change to the description (rebuild when it moves).
    revision: u64,
}

impl Editor {
    pub fn new(desc: SceneDesc) -> Self {
        Self {
            desc,
            selected: None,
            undo: Vec::new(),
            redo: Vec::new(),
            pending: None,
            revision: 0,
        }
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn desc(&self) -> &SceneDesc {
        &self.desc
    }

    pub fn selected(&self) -> Option<usize> {
        self.selected
    }

    pub fn select(&mut self, index: Option<usize>) {
        self.selected = index.filter(|&i| i < self.desc.objects.len());
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty() || self.pending.is_some()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Starts a gesture: later `modify` calls share one undo step.
    pub fn begin(&mut self) {
        if self.pending.is_none() {
            self.pending = Some(self.desc.clone());
        }
    }

    /// Changes the description as part of the current gesture.
    pub fn modify(&mut self, change: impl FnOnce(&mut SceneDesc)) {
        self.begin();
        change(&mut self.desc);
        self.revision += 1;
        self.select(self.selected);
    }

    /// Ends the gesture; records an undo step (and returns true) if anything
    /// changed.
    pub fn commit(&mut self) -> bool {
        match self.pending.take() {
            Some(before) if before != self.desc => {
                self.push_undo(before);
                self.redo.clear();
                true
            }
            _ => false,
        }
    }

    /// One complete edit (its own undo step). True if anything changed.
    pub fn edit(&mut self, change: impl FnOnce(&mut SceneDesc)) -> bool {
        self.commit();
        self.modify(change);
        self.commit()
    }

    fn push_undo(&mut self, snapshot: SceneDesc) {
        if self.undo.len() == UNDO_DEPTH {
            self.undo.remove(0);
        }
        self.undo.push(snapshot);
    }

    pub fn undo(&mut self) -> bool {
        self.commit();
        let Some(previous) = self.undo.pop() else {
            return false;
        };
        self.redo.push(std::mem::replace(&mut self.desc, previous));
        self.revision += 1;
        self.select(self.selected);
        true
    }

    pub fn redo(&mut self) -> bool {
        self.commit();
        let Some(next) = self.redo.pop() else {
            return false;
        };
        let previous = std::mem::replace(&mut self.desc, next);
        self.push_undo(previous);
        self.revision += 1;
        self.select(self.selected);
        true
    }

    /// A different scene (loaded or picked): fresh history, nothing selected.
    pub fn replace(&mut self, desc: SceneDesc) {
        let revision = self.revision + 1;
        *self = Self::new(desc);
        self.revision = revision;
    }

    /// Adds an object and selects it; returns its index.
    pub fn add(&mut self, object: ObjectDesc) -> usize {
        self.edit(|d| d.objects.push(object));
        let index = self.desc.objects.len() - 1;
        self.selected = Some(index);
        index
    }

    /// Copies the selection and everything under it beside the original, with
    /// the joints wholly inside that subtree; selects the copy.
    pub fn duplicate_selected(&mut self) -> Option<usize> {
        let root = self.selected?;
        let ids = subtree(&self.desc, root);
        let first = self.desc.objects.len();
        let new_index = |old: usize| ids.iter().position(|&i| i == old).map(|k| first + k);
        let mut copies = Vec::with_capacity(ids.len());
        for &i in &ids {
            let mut copy = self.desc.objects[i].clone();
            if i == root {
                // Offset in the parent's frame; the copy stays the original's sibling.
                copy.position = (Vec3::from_array(copy.position) + DUPLICATE_OFFSET).to_array();
                copy.name = format!("{} copy", copy.name);
            } else {
                copy.parent = copy.parent.and_then(new_index);
            }
            copies.push(copy);
        }
        let joints: Vec<JointSpec> = self
            .desc
            .joints
            .iter()
            .filter_map(|j| {
                let (a, b) = (new_index(j.a)?, new_index(j.b)?);
                Some(JointSpec { a, b, ..j.clone() })
            })
            .collect();
        self.edit(|d| {
            d.objects.extend(copies);
            d.joints.extend(joints);
        });
        self.selected = Some(first);
        Some(first)
    }

    /// Deletes the selection, everything under it, and every joint touching
    /// them; remaining parents and joints are renumbered.
    pub fn delete_selected(&mut self) -> bool {
        let Some(index) = self.selected else {
            return false;
        };
        let mut remove = vec![false; self.desc.objects.len()];
        for i in subtree(&self.desc, index) {
            remove[i] = true;
        }
        self.edit(|d| remove_objects(d, &remove));
        self.selected = None;
        true
    }

    /// Moves `child` under `parent` (or to the top level), keeping where it is
    /// in the world. Refused if `parent` is `child` or one of its descendants.
    /// True if anything changed.
    pub fn reparent(&mut self, child: usize, parent: Option<usize>) -> Result<bool, ReparentError> {
        let n = self.desc.objects.len();
        if child >= n || parent.is_some_and(|p| p >= n) {
            return Err(ReparentError::NoSuchObject);
        }
        if parent.is_some_and(|p| is_within(&self.desc, p, child)) {
            return Err(ReparentError::IntoOwnSubtree);
        }
        if self.desc.objects[child].parent == parent {
            return Ok(false);
        }
        world_poses(&self.desc).map_err(|_| ReparentError::NoSuchObject)?;
        Ok(self.edit(|d| {
            set_parent_keep_world(d, child, parent);
        }))
    }

    /// Writes simulated poses back as authored ones (velocities zeroed: a
    /// captured pose is a starting rest state). Objects with joints keep their
    /// authored pose, since joints are defined at it. Poses are world space
    /// (`poses[i]` for object `i`, `None` if unknown) and are stored relative
    /// to each object's parent as the parent now stands. Returns how many
    /// objects were captured.
    pub fn capture_poses(&mut self, poses: &[Option<(Vec3, Quat)>]) -> usize {
        let Ok(authored) = world_poses(&self.desc) else {
            return 0;
        };
        let n = self.desc.objects.len();
        let jointed: Vec<bool> = (0..n)
            .map(|i| self.desc.joints.iter().any(|j| j.a == i || j.b == i))
            .collect();
        let captured: Vec<Option<(Vec3, Quat)>> = (0..n)
            .map(|i| poses.get(i).copied().flatten().filter(|_| !jointed[i]))
            .collect();
        let now: Vec<(Vec3, Quat)> = (0..n).map(|i| captured[i].unwrap_or(authored[i])).collect();
        let count = captured.iter().flatten().count();
        self.edit(|d| {
            for (i, object) in d.objects.iter_mut().enumerate() {
                let Some((position, rotation)) = captured[i] else {
                    continue;
                };
                let (position, rotation) = match object.parent {
                    Some(p) => {
                        let (pp, pq) = now[p];
                        (pq.inverse() * (position - pp), pq.inverse() * rotation)
                    }
                    None => (position, rotation),
                };
                object.position = position.to_array();
                object.set_orientation(rotation);
                object.velocity = [0.0; 3];
                object.angular_velocity = [0.0; 3];
            }
        });
        count
    }

    /// Adds an empty (no shape, static: a folder for the hierarchy, or a joint
    /// anchor) at `at` and selects it.
    pub fn add_empty(&mut self, at: Vec3) -> usize {
        let index = self.desc.objects.len();
        self.add(ObjectDesc {
            name: format!("empty {index}"),
            shape: None,
            position: at.to_array(),
            body: BodyType::Static,
            ..ObjectDesc::default()
        })
    }
}

/// Why a reparent was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReparentError {
    NoSuchObject,
    /// The new parent is the object itself or one of its descendants.
    IntoOwnSubtree,
}

/// Whether `i` is `root` or one of its descendants (tolerates bad parents).
pub fn is_within(desc: &SceneDesc, mut i: usize, root: usize) -> bool {
    for _ in 0..=desc.objects.len() {
        if i == root {
            return true;
        }
        match desc.objects.get(i).and_then(|o| o.parent) {
            Some(p) => i = p,
            None => return false,
        }
    }
    false
}

/// `root` and everything under it, in list order.
pub fn subtree(desc: &SceneDesc, root: usize) -> Vec<usize> {
    (0..desc.objects.len())
        .filter(|&i| is_within(desc, i, root))
        .collect()
}

/// Removes the flagged objects and every joint touching one, renumbering
/// the remaining parents and joints. A kept object whose parent goes becomes
/// top-level where it stands in the world.
pub fn remove_objects(desc: &mut SceneDesc, remove: &[bool]) {
    let world = world_poses(desc).ok();
    let mut new_index = vec![None; desc.objects.len()];
    let mut next = 0;
    for (i, slot) in new_index.iter_mut().enumerate() {
        if !remove.get(i).copied().unwrap_or(false) {
            *slot = Some(next);
            next += 1;
        }
    }
    let old = std::mem::take(&mut desc.objects);
    for (i, mut object) in old.into_iter().enumerate() {
        if new_index[i].is_none() {
            continue;
        }
        if let Some(p) = object.parent {
            match new_index.get(p).copied().flatten() {
                Some(np) => object.parent = Some(np),
                None => {
                    object.parent = None;
                    if let Some(world) = &world {
                        object.position = world[i].0.to_array();
                        object.set_orientation(world[i].1);
                    }
                }
            }
        }
        desc.objects.push(object);
    }
    desc.joints.retain_mut(|j| {
        match (
            new_index.get(j.a).copied().flatten(),
            new_index.get(j.b).copied().flatten(),
        ) {
            (Some(a), Some(b)) => {
                j.a = a;
                j.b = b;
                true
            }
            _ => false,
        }
    });
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_desc::{Arena, SCENE_VERSION, ShapeDesc};

    fn ball(x: f32) -> ObjectDesc {
        ObjectDesc {
            name: format!("ball {x}"),
            shape: Some(ShapeDesc::Sphere { radius: 0.5 }),
            position: [x, 1.0, 0.0],
            ..ObjectDesc::default()
        }
    }

    /// Five balls; joints 0–1, 2–3, 1–3.
    fn scene() -> SceneDesc {
        SceneDesc {
            version: SCENE_VERSION,
            name: "test".into(),
            arena: Arena::Floor,
            objects: (0..5).map(|i| ball(i as f32)).collect(),
            joints: [(0, 1), (2, 3), (1, 3)]
                .map(|(a, b)| JointSpec::ball_socket(a, b, Vec3::ZERO))
                .to_vec(),
        }
    }

    #[test]
    fn add_selects_and_undo_redo_restore_exactly() {
        let mut e = Editor::new(scene());
        let original = e.desc().clone();
        let i = e.add(ball(9.0));
        assert_eq!((i, e.selected()), (5, Some(5)));
        let added = e.desc().clone();
        assert!(e.undo());
        assert_eq!(e.desc(), &original);
        assert_eq!(e.selected(), None, "selection past the end is cleared");
        assert!(e.redo());
        assert_eq!(e.desc(), &added);
        assert!(!e.redo());
    }

    #[test]
    fn delete_removes_its_joints_and_renumbers_the_rest() {
        let mut e = Editor::new(scene());
        e.select(Some(1));
        assert!(e.delete_selected());
        let d = e.desc();
        assert_eq!(d.objects.len(), 4);
        assert_eq!(d.objects[1].name, "ball 2");
        let pairs: Vec<_> = d.joints.iter().map(|j| (j.a, j.b)).collect();
        assert_eq!(pairs, [(1, 2)], "2–3 renumbered; joints on 1 gone");
        assert_eq!(e.selected(), None);
        assert!(e.undo());
        assert_eq!(e.desc(), &scene());
    }

    #[test]
    fn duplicate_copies_the_object_but_not_its_joints() {
        let mut e = Editor::new(scene());
        e.select(Some(3));
        let copy = e.duplicate_selected().unwrap();
        let d = e.desc();
        assert_eq!(copy, 5);
        assert_eq!(d.objects[5].shape, d.objects[3].shape);
        assert_eq!(d.objects[5].position, [3.5, 1.0, 0.5]);
        assert_eq!(d.joints.len(), 3);
        assert_eq!(e.selected(), Some(5));
        e.select(None);
        assert_eq!(e.duplicate_selected(), None);
    }

    #[test]
    fn a_gesture_is_one_undo_step_and_no_op_edits_record_nothing() {
        let mut e = Editor::new(scene());
        assert!(!e.edit(|_| {}));
        assert!(!e.can_undo());
        for step in 1..=10 {
            e.modify(|d| d.objects[0].position[1] = step as f32);
        }
        e.commit();
        assert_eq!(e.desc().objects[0].position[1], 10.0);
        assert!(e.undo());
        assert_eq!(e.desc(), &scene(), "the whole drag undoes at once");
        assert!(!e.can_undo());
    }

    #[test]
    fn undo_history_is_capped() {
        let mut e = Editor::new(scene());
        for i in 0..UNDO_DEPTH + 20 {
            e.edit(|d| d.objects[0].position[0] = i as f32 + 100.0);
        }
        let mut undone = 0;
        while e.undo() {
            undone += 1;
        }
        assert_eq!(undone, UNDO_DEPTH);
    }

    #[test]
    fn an_edit_clears_redo() {
        let mut e = Editor::new(scene());
        e.add(ball(7.0));
        e.undo();
        e.add(ball(8.0));
        assert!(!e.can_redo());
    }

    #[test]
    fn capture_writes_unjointed_poses_and_zeroes_their_velocity() {
        let mut desc = scene();
        desc.objects[4].velocity = [1.0, 2.0, 3.0];
        let mut e = Editor::new(desc);
        let spun = Quat::from_rotation_y(0.5);
        let poses: Vec<_> = (0..5)
            .map(|i| Some((Vec3::new(i as f32, 9.0, 0.0), spun)))
            .collect();
        assert_eq!(e.capture_poses(&poses), 1, "only object 4 has no joints");
        let d = e.desc();
        assert_eq!(d.objects[4].position, [4.0, 9.0, 0.0]);
        assert!(d.objects[4].orientation().abs_diff_eq(spun, 1e-5));
        assert_eq!(d.objects[4].velocity, [0.0; 3]);
        assert_eq!(d.objects[0].position, [0.0, 1.0, 0.0], "jointed: kept");
        assert!(e.undo());
        assert_eq!(e.desc(), &scene_with_velocity());

        fn scene_with_velocity() -> SceneDesc {
            let mut d = scene();
            d.objects[4].velocity = [1.0, 2.0, 3.0];
            d
        }
    }

    fn node(name: &str, parent: Option<usize>, position: [f32; 3]) -> ObjectDesc {
        ObjectDesc {
            name: name.into(),
            parent,
            position,
            shape: Some(ShapeDesc::Sphere { radius: 0.1 }),
            ..ObjectDesc::default()
        }
    }

    /// A(B(C)), D(E); joints A–D, C–E, D–E.
    fn tree() -> SceneDesc {
        SceneDesc {
            version: SCENE_VERSION,
            name: "tree".into(),
            arena: Arena::Empty,
            objects: vec![
                node("A", None, [0.0, 1.0, 0.0]),
                node("B", Some(0), [1.0, 0.0, 0.0]),
                node("C", Some(1), [1.0, 0.0, 0.0]),
                node("D", None, [5.0, 1.0, 0.0]),
                node("E", Some(3), [0.0, 1.0, 0.0]),
            ],
            joints: [(0, 3), (2, 4), (3, 4)]
                .map(|(a, b)| JointSpec::ball_socket(a, b, Vec3::ZERO))
                .to_vec(),
        }
    }

    fn world(e: &Editor) -> Vec<(Vec3, Quat)> {
        world_poses(e.desc()).unwrap()
    }

    #[test]
    fn deleting_takes_the_subtree_and_renumbers_parents_and_joints() {
        let mut e = Editor::new(tree());
        e.select(Some(0));
        assert!(e.delete_selected());
        let d = e.desc();
        let names: Vec<_> = d.objects.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(names, ["D", "E"]);
        assert_eq!(d.objects[1].parent, Some(0), "E's parent D renumbered");
        let pairs: Vec<_> = d.joints.iter().map(|j| (j.a, j.b)).collect();
        assert_eq!(pairs, [(0, 1)], "only D–E survives");
        assert!(e.undo());
        assert_eq!(e.desc(), &tree(), "one undo step");
    }

    #[test]
    fn duplicating_copies_the_subtree_and_its_inner_joints() {
        let mut desc = tree();
        desc.joints.push(JointSpec::ball_socket(0, 1, Vec3::ZERO)); // inside A's subtree
        let mut e = Editor::new(desc);
        e.select(Some(0));
        let copy = e.duplicate_selected().unwrap();
        let d = e.desc();
        assert_eq!(copy, 5);
        assert_eq!(d.objects.len(), 8, "A, B, C copied");
        assert_eq!(d.objects[5].name, "A copy");
        assert_eq!(d.objects[5].parent, None);
        assert_eq!(
            (d.objects[6].parent, d.objects[7].parent),
            (Some(5), Some(6))
        );
        let pairs: Vec<_> = d.joints.iter().map(|j| (j.a, j.b)).collect();
        assert_eq!(
            pairs,
            [(0, 3), (2, 4), (3, 4), (0, 1), (5, 6)],
            "only the inner joint copied"
        );
        // The copied subtree keeps its shape, offset as a whole.
        let w = world(&e);
        assert!((w[7].0 - w[5].0).abs_diff_eq(w[2].0 - w[0].0, 1e-5));
        assert!(e.undo());
        assert_eq!(e.desc().objects.len(), 5);
    }

    #[test]
    fn reparenting_keeps_world_poses_and_refuses_loops() {
        let mut desc = tree();
        desc.objects[3].rotation_deg = [0.0, 90.0, 0.0];
        let mut e = Editor::new(desc);
        let before = world(&e);
        assert_eq!(e.reparent(2, Some(3)), Ok(true), "C under the turned D");
        let after = world(&e);
        assert!(after[2].0.abs_diff_eq(before[2].0, 1e-5));
        assert!(after[2].1.abs_diff_eq(before[2].1, 1e-5));
        assert_eq!(e.desc().objects[2].parent, Some(3));
        assert_eq!(e.reparent(2, None), Ok(true), "back to the top level");
        assert!(world(&e)[2].0.abs_diff_eq(before[2].0, 1e-5));

        assert_eq!(e.reparent(0, Some(1)), Err(ReparentError::IntoOwnSubtree));
        assert_eq!(e.reparent(0, Some(0)), Err(ReparentError::IntoOwnSubtree));
        assert_eq!(e.reparent(0, Some(99)), Err(ReparentError::NoSuchObject));
        assert_eq!(e.reparent(1, Some(0)), Ok(false), "already there");
        assert!(e.undo() && e.undo());
        assert_eq!(e.desc().objects[2].parent, Some(1));
    }

    #[test]
    fn captured_children_stay_relative_to_their_parents() {
        let mut desc = tree();
        desc.joints.clear();
        let mut e = Editor::new(desc);
        let turned = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let mut poses: Vec<Option<(Vec3, Quat)>> = vec![None; 5];
        poses[0] = Some((Vec3::new(10.0, 0.0, 0.0), turned));
        poses[1] = Some((Vec3::new(10.0, 0.0, -1.0), turned));
        assert_eq!(e.capture_poses(&poses), 2);
        let d = e.desc();
        assert_eq!(d.objects[1].parent, Some(0));
        // B sits 1 m along A's local +X (world -Z once A is turned).
        assert!(Vec3::from_array(d.objects[1].position).abs_diff_eq(Vec3::X, 1e-5));
        assert!(world(&e)[1].0.abs_diff_eq(Vec3::new(10.0, 0.0, -1.0), 1e-5));
    }

    #[test]
    fn replace_starts_fresh() {
        let mut e = Editor::new(scene());
        e.add(ball(7.0));
        e.replace(scene());
        assert!(!e.can_undo() && !e.can_redo());
        assert_eq!(e.selected(), None);
    }
}
