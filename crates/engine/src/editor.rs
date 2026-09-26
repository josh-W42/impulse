//! Editor state with no UI: the authored scene, the selection, and undo/redo.
//! Undo stores whole-description snapshots (scenes are small data).
//!
//! Continuous edits (dragging a value or a body) are one undo step: `begin`,
//! any number of `modify`, then `commit`. `edit` does all three at once.

use glam::{Quat, Vec3};

use crate::scene_desc::{ObjectDesc, SceneDesc};

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

    /// Copies the selection (not its joints) beside the original and selects
    /// the copy.
    pub fn duplicate_selected(&mut self) -> Option<usize> {
        let original = self.desc.objects.get(self.selected?)?;
        let mut copy = original.clone();
        copy.position = (Vec3::from_array(copy.position) + DUPLICATE_OFFSET).to_array();
        copy.name = format!("{} copy", copy.name);
        Some(self.add(copy))
    }

    /// Deletes the selection and its joints; other joints are renumbered.
    pub fn delete_selected(&mut self) -> bool {
        let Some(index) = self.selected else {
            return false;
        };
        self.edit(|d| {
            d.objects.remove(index);
            d.joints.retain(|j| j.a != index && j.b != index);
            for j in &mut d.joints {
                j.a -= usize::from(j.a > index);
                j.b -= usize::from(j.b > index);
            }
        });
        self.selected = None;
        true
    }

    /// Writes simulated poses back as authored ones (velocities zeroed: a
    /// captured pose is a starting rest state). Objects with joints keep their
    /// authored pose, since joints are defined at it. `poses[i]` is object
    /// `i`'s pose, `None` if unknown. Returns how many objects were captured.
    pub fn capture_poses(&mut self, poses: &[Option<(Vec3, Quat)>]) -> usize {
        let jointed: Vec<bool> = (0..self.desc.objects.len())
            .map(|i| self.desc.joints.iter().any(|j| j.a == i || j.b == i))
            .collect();
        let mut captured = 0;
        self.edit(|d| {
            for (i, object) in d.objects.iter_mut().enumerate() {
                if let Some(Some((position, orientation))) = poses.get(i)
                    && !jointed[i]
                {
                    object.position = position.to_array();
                    object.set_orientation(*orientation);
                    object.velocity = [0.0; 3];
                    object.angular_velocity = [0.0; 3];
                    captured += 1;
                }
            }
        });
        captured
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_desc::{Arena, JointSpec, SCENE_VERSION, ShapeDesc};

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

    #[test]
    fn replace_starts_fresh() {
        let mut e = Editor::new(scene());
        e.add(ball(7.0));
        e.replace(scene());
        assert!(!e.can_undo() && !e.can_redo());
        assert_eq!(e.selected(), None);
    }
}
