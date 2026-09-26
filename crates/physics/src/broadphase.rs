//! Broadphase: a dynamic AABB tree (Box2D v2 lineage). Leaves hold "fat" boxes so
//! bodies moving within their margin need no tree update; insertion picks a sibling
//! by surface-area cost; AVL-style rotations keep the tree balanced.

use glam::{Mat3, Quat, Vec3};

use crate::collider::Collider;
use crate::world::BodyHandle;

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// World bounds of a collider; `None` for unbounded shapes (planes).
    pub fn of(collider: Collider, position: Vec3, orientation: Quat) -> Option<Aabb> {
        let extent = match collider {
            Collider::Sphere { radius } => Vec3::splat(radius),
            // Projection of the box onto each world axis: |R|·h.
            Collider::Box { half_extents } => Mat3::from_quat(orientation).abs() * half_extents,
            Collider::Capsule {
                half_height,
                radius,
            } => (orientation * Vec3::Y * half_height).abs() + Vec3::splat(radius),
            Collider::Plane { .. } => return None,
        };
        Some(Aabb {
            min: position - extent,
            max: position + extent,
        })
    }

    pub fn union(self, other: Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn overlaps(self, other: Aabb) -> bool {
        self.min.cmple(other.max).all() && other.min.cmple(self.max).all()
    }

    pub fn contains(self, other: Aabb) -> bool {
        self.min.cmple(other.min).all() && other.max.cmple(self.max).all()
    }

    /// Surface area: the insertion cost heuristic.
    fn area(self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    fn fattened(self, margin: f32) -> Aabb {
        Aabb {
            min: self.min - Vec3::splat(margin),
            max: self.max + Vec3::splat(margin),
        }
    }
}

const NULL: usize = usize::MAX;
/// How far a body may move before its leaf is re-inserted.
const MARGIN: f32 = 0.1;

#[derive(Debug, Clone)]
struct Node {
    aabb: Aabb,
    parent: usize,
    children: [usize; 2],
    /// Leaves are 0; free nodes are -1.
    height: i32,
    body: Option<BodyHandle>,
}

impl Node {
    fn is_leaf(&self) -> bool {
        self.children[0] == NULL
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AabbTree {
    nodes: Vec<Node>,
    root: Option<usize>,
    free: Vec<usize>,
}

impl AabbTree {
    fn allocate(&mut self, node: Node) -> usize {
        if let Some(i) = self.free.pop() {
            self.nodes[i] = node;
            i
        } else {
            self.nodes.push(node);
            self.nodes.len() - 1
        }
    }

    fn release(&mut self, i: usize) {
        self.nodes[i].height = -1;
        self.nodes[i].body = None;
        self.free.push(i);
    }

    /// Adds a body's bounds; returns its proxy id.
    pub(crate) fn insert(&mut self, aabb: Aabb, body: BodyHandle) -> usize {
        let leaf = self.allocate(Node {
            aabb: aabb.fattened(MARGIN),
            parent: NULL,
            children: [NULL, NULL],
            height: 0,
            body: Some(body),
        });
        self.insert_leaf(leaf);
        leaf
    }

    pub(crate) fn remove(&mut self, proxy: usize) {
        self.remove_leaf(proxy);
        self.release(proxy);
    }

    /// Updates a proxy's bounds; re-inserts only if it left its fat box.
    pub(crate) fn update(&mut self, proxy: usize, aabb: Aabb) {
        if self.nodes[proxy].aabb.contains(aabb) {
            return;
        }
        self.remove_leaf(proxy);
        self.nodes[proxy].aabb = aabb.fattened(MARGIN);
        self.insert_leaf(proxy);
    }

    pub(crate) fn fat_aabb(&self, proxy: usize) -> Aabb {
        self.nodes[proxy].aabb
    }

    fn insert_leaf(&mut self, leaf: usize) {
        let Some(root) = self.root else {
            self.root = Some(leaf);
            self.nodes[leaf].parent = NULL;
            return;
        };
        let leaf_aabb = self.nodes[leaf].aabb;

        // Descend toward the sibling with the least surface-area cost.
        let mut index = root;
        while !self.nodes[index].is_leaf() {
            let node = &self.nodes[index];
            let area = node.aabb.area();
            let combined = node.aabb.union(leaf_aabb).area();
            let cost_here = 2.0 * combined;
            let inheritance = 2.0 * (combined - area);
            let child_cost = |c: usize| {
                let child = &self.nodes[c];
                let enlarged = leaf_aabb.union(child.aabb).area();
                let growth = if child.is_leaf() {
                    enlarged
                } else {
                    enlarged - child.aabb.area()
                };
                growth + inheritance
            };
            let [c0, c1] = node.children;
            let (cost0, cost1) = (child_cost(c0), child_cost(c1));
            if cost_here < cost0 && cost_here < cost1 {
                break;
            }
            index = if cost0 < cost1 { c0 } else { c1 };
        }

        let sibling = index;
        let old_parent = self.nodes[sibling].parent;
        let new_parent = self.allocate(Node {
            aabb: leaf_aabb.union(self.nodes[sibling].aabb),
            parent: old_parent,
            children: [sibling, leaf],
            height: self.nodes[sibling].height + 1,
            body: None,
        });
        if old_parent == NULL {
            self.root = Some(new_parent);
        } else {
            self.replace_child(old_parent, sibling, new_parent);
        }
        self.nodes[sibling].parent = new_parent;
        self.nodes[leaf].parent = new_parent;
        self.refit_upward(new_parent);
    }

    fn remove_leaf(&mut self, leaf: usize) {
        if self.root == Some(leaf) {
            self.root = None;
            return;
        }
        let parent = self.nodes[leaf].parent;
        let grandparent = self.nodes[parent].parent;
        let [c0, c1] = self.nodes[parent].children;
        let sibling = if c0 == leaf { c1 } else { c0 };
        if grandparent == NULL {
            self.root = Some(sibling);
            self.nodes[sibling].parent = NULL;
        } else {
            self.replace_child(grandparent, parent, sibling);
            self.nodes[sibling].parent = grandparent;
            self.refit_upward(grandparent);
        }
        self.release(parent);
    }

    fn replace_child(&mut self, parent: usize, old: usize, new: usize) {
        let children = &mut self.nodes[parent].children;
        if children[0] == old {
            children[0] = new;
        } else {
            children[1] = new;
        }
    }

    /// Rebalances and refits bounds from `index` to the root.
    fn refit_upward(&mut self, mut index: usize) {
        while index != NULL {
            index = self.balance(index);
            let [c0, c1] = self.nodes[index].children;
            self.nodes[index].height = 1 + self.nodes[c0].height.max(self.nodes[c1].height);
            self.nodes[index].aabb = self.nodes[c0].aabb.union(self.nodes[c1].aabb);
            index = self.nodes[index].parent;
        }
    }

    /// If `a`'s subtrees differ in height by more than one, rotate the taller
    /// child up (AVL-style). Returns the index now at `a`'s position.
    fn balance(&mut self, a: usize) -> usize {
        if self.nodes[a].is_leaf() || self.nodes[a].height < 2 {
            return a;
        }
        let [b, c] = self.nodes[a].children;
        let lean = self.nodes[c].height - self.nodes[b].height;
        if lean > 1 {
            self.rotate_up(a, c, b, 1)
        } else if lean < -1 {
            self.rotate_up(a, b, c, 0)
        } else {
            a
        }
    }

    /// Promotes `up` (child `slot` of `a`) above `a`; `other` is `a`'s other child.
    fn rotate_up(&mut self, a: usize, up: usize, other: usize, slot: usize) -> usize {
        let [f, g] = self.nodes[up].children;
        let a_parent = self.nodes[a].parent;
        self.nodes[up].children[0] = a;
        self.nodes[up].parent = a_parent;
        self.nodes[a].parent = up;
        if a_parent == NULL {
            self.root = Some(up);
        } else {
            self.replace_child(a_parent, a, up);
        }
        // `up` keeps its taller grandchild; the shorter one moves under `a`.
        let (keep, give) = if self.nodes[f].height > self.nodes[g].height {
            (f, g)
        } else {
            (g, f)
        };
        self.nodes[up].children[1] = keep;
        self.nodes[a].children[slot] = give;
        self.nodes[give].parent = a;
        self.nodes[a].aabb = self.nodes[other].aabb.union(self.nodes[give].aabb);
        self.nodes[a].height = 1 + self.nodes[other].height.max(self.nodes[give].height);
        self.nodes[up].aabb = self.nodes[a].aabb.union(self.nodes[keep].aabb);
        self.nodes[up].height = 1 + self.nodes[a].height.max(self.nodes[keep].height);
        up
    }

    /// Calls `visit` for every leaf whose fat box overlaps `aabb`.
    pub(crate) fn query(&self, aabb: Aabb, mut visit: impl FnMut(BodyHandle)) {
        let Some(root) = self.root else {
            return;
        };
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            let node = &self.nodes[i];
            if !node.aabb.overlaps(aabb) {
                continue;
            }
            if let Some(body) = node.body {
                visit(body);
            } else {
                stack.extend(node.children);
            }
        }
    }

    /// Tree height (0 for a single leaf, -1 when empty).
    #[cfg(test)]
    pub(crate) fn height(&self) -> i32 {
        self.root.map_or(-1, |r| self.nodes[r].height)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use super::*;
    use crate::body::BodyDesc;
    use crate::world::World;

    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 8) as f32 / (1u32 << 24) as f32
        }
        fn aabb(&mut self) -> Aabb {
            let center = Vec3::new(self.next(), self.next(), self.next()) * 20.0;
            let half = Vec3::new(self.next(), self.next(), self.next()) * 0.8 + 0.1;
            Aabb {
                min: center - half,
                max: center + half,
            }
        }
    }

    /// Real handles to label leaves (the tree only stores them).
    fn handles(n: usize) -> Vec<BodyHandle> {
        let mut world = World::default();
        (0..n)
            .map(|_| world.add_body(BodyDesc::dynamic(1.0, Vec3::ZERO)).unwrap())
            .collect()
    }

    fn tree_pairs(
        tree: &AabbTree,
        fat: &HashMap<BodyHandle, Aabb>,
    ) -> BTreeSet<(BodyHandle, BodyHandle)> {
        let mut pairs = BTreeSet::new();
        for (&h, &aabb) in fat {
            tree.query(aabb, |other| {
                if other != h {
                    pairs.insert((h.min(other), h.max(other)));
                }
            });
        }
        pairs
    }

    fn brute_pairs(fat: &HashMap<BodyHandle, Aabb>) -> BTreeSet<(BodyHandle, BodyHandle)> {
        let mut pairs = BTreeSet::new();
        for (&a, &aa) in fat {
            for (&b, &bb) in fat {
                if a < b && aa.overlaps(bb) {
                    pairs.insert((a, b));
                }
            }
        }
        pairs
    }

    #[test]
    fn aabbs_bound_their_shapes() {
        let sphere = Aabb::of(Collider::Sphere { radius: 0.5 }, Vec3::ONE, Quat::IDENTITY).unwrap();
        assert_eq!(
            sphere,
            Aabb {
                min: Vec3::splat(0.5),
                max: Vec3::splat(1.5)
            }
        );
        // A unit cube turned 45° about y is √2 wide in x and z.
        let turned = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        let cube = Aabb::of(
            Collider::Box {
                half_extents: Vec3::splat(0.5),
            },
            Vec3::ZERO,
            turned,
        )
        .unwrap();
        assert!((cube.max.x - 0.5 * 2f32.sqrt()).abs() < 1e-5);
        assert!((cube.max.y - 0.5).abs() < 1e-5);
        assert!(
            Aabb::of(
                Collider::Plane { normal: Vec3::Y },
                Vec3::ZERO,
                Quat::IDENTITY
            )
            .is_none()
        );
    }

    #[test]
    fn pairs_match_brute_force_through_inserts_moves_and_removals() {
        let mut rng = Rng(9);
        let hs = handles(300);
        let mut tree = AabbTree::default();
        let mut proxies = HashMap::new();
        for &h in &hs {
            proxies.insert(h, tree.insert(rng.aabb(), h));
        }
        let fat = |tree: &AabbTree, proxies: &HashMap<BodyHandle, usize>| {
            proxies
                .iter()
                .map(|(&h, &p)| (h, tree.fat_aabb(p)))
                .collect::<HashMap<_, _>>()
        };
        assert_eq!(
            tree_pairs(&tree, &fat(&tree, &proxies)),
            brute_pairs(&fat(&tree, &proxies))
        );

        for (i, &h) in hs.iter().enumerate() {
            if i % 3 == 0 {
                tree.update(proxies[&h], rng.aabb());
            }
        }
        for &h in hs.iter().step_by(5) {
            tree.remove(proxies.remove(&h).unwrap());
        }
        let f = fat(&tree, &proxies);
        assert_eq!(tree_pairs(&tree, &f), brute_pairs(&f));
    }

    #[test]
    fn tree_stays_balanced_even_for_sorted_insertion() {
        let hs = handles(1024);
        let mut tree = AabbTree::default();
        for (i, &h) in hs.iter().enumerate() {
            // Sorted along x: the worst case for naive insertion (a chain).
            let min = Vec3::new(i as f32 * 1.5, 0.0, 0.0);
            tree.insert(
                Aabb {
                    min,
                    max: min + Vec3::ONE,
                },
                h,
            );
        }
        // AVL bound ≈ 1.44·log2(n) = 14.4 for 1024 leaves.
        assert!(tree.height() <= 15, "height {}", tree.height());
    }

    #[test]
    fn small_moves_inside_the_margin_do_not_touch_the_tree() {
        let hs = handles(1);
        let mut tree = AabbTree::default();
        let aabb = Aabb {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        };
        let p = tree.insert(aabb, hs[0]);
        let before = tree.fat_aabb(p);
        let nudge = Vec3::splat(MARGIN * 0.5);
        tree.update(
            p,
            Aabb {
                min: aabb.min + nudge,
                max: aabb.max + nudge,
            },
        );
        assert_eq!(tree.fat_aabb(p), before);
    }
}
