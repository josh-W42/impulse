//! Disjoint sets over body slot indices, for grouping touching bodies into islands.

pub(crate) struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    pub(crate) fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
        }
    }

    pub(crate) fn find(&mut self, mut i: usize) -> usize {
        while self.parent[i] != i {
            self.parent[i] = self.parent[self.parent[i]]; // path halving
            i = self.parent[i];
        }
        i
    }

    pub(crate) fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // Lower index as root: deterministic regardless of contact order.
            let (root, child) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.parent[child] = root;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unions_are_transitive_and_separate_sets_stay_apart() {
        let mut uf = UnionFind::new(6);
        uf.union(0, 1);
        uf.union(3, 1);
        uf.union(4, 5);
        assert_eq!(uf.find(3), uf.find(0));
        assert_eq!(uf.find(5), uf.find(4));
        assert_ne!(uf.find(0), uf.find(4));
        assert_eq!(uf.find(2), 2);
    }
}
