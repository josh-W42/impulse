//! Collision filtering: which bodies collide with each other, and which bodies a
//! query (raycast, shape cast, overlap) can see.

use crate::world::BodyHandle;

/// Collision groups as bit masks. A body belongs to the groups set in `groups`
/// and collides with bodies in the groups set in `mask`; two bodies collide
/// only if each one's mask accepts the other's groups.
///
/// The default puts a body in group 1 and lets it collide with every group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CollisionFilter {
    pub groups: u32,
    pub mask: u32,
}

impl CollisionFilter {
    /// Group 1, colliding with everything.
    pub const DEFAULT: Self = Self {
        groups: 1,
        mask: u32::MAX,
    };

    pub const fn new(groups: u32, mask: u32) -> Self {
        Self { groups, mask }
    }

    /// Both masks must accept: `a` collides with `b` exactly when `b` collides with `a`.
    pub fn collides_with(self, other: Self) -> bool {
        self.mask & other.groups != 0 && other.mask & self.groups != 0
    }
}

impl Default for CollisionFilter {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Which bodies a query sees: those in a group set in `mask`, minus `exclude`
/// (typically the body doing the querying, such as a character's own capsule).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryFilter {
    pub mask: u32,
    pub exclude: Option<BodyHandle>,
}

impl QueryFilter {
    /// Every body.
    pub const ALL: Self = Self {
        mask: u32::MAX,
        exclude: None,
    };

    /// Bodies in any group set in `mask`.
    pub const fn mask(mask: u32) -> Self {
        Self {
            mask,
            exclude: None,
        }
    }

    /// The same filter, also skipping `body`.
    pub const fn excluding(self, body: BodyHandle) -> Self {
        Self {
            exclude: Some(body),
            ..self
        }
    }

    pub(crate) fn accepts(self, handle: BodyHandle, filter: CollisionFilter) -> bool {
        self.mask & filter.groups != 0 && self.exclude != Some(handle)
    }
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self::ALL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_masks_must_accept() {
        let player = CollisionFilter::new(0b001, 0b110);
        let wall = CollisionFilter::new(0b010, u32::MAX);
        let ghost = CollisionFilter::new(0b100, 0b010);
        assert!(player.collides_with(wall) && wall.collides_with(player));
        // The player's mask accepts ghosts, but the ghost's mask rejects players.
        assert!(!player.collides_with(ghost) && !ghost.collides_with(player));
        assert!(ghost.collides_with(wall));
        assert!(CollisionFilter::DEFAULT.collides_with(CollisionFilter::DEFAULT));
    }
}
