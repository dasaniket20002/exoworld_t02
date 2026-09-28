use parking_lot::{RwLock, RwLockReadGuard};

use crate::entities::entity_id::EntityId;

pub struct GridCell {
    entities: RwLock<Vec<EntityId>>,
}

impl GridCell {
    pub fn new() -> Self {
        Self {
            entities: RwLock::new(Vec::with_capacity(8)),
        }
    }

    pub fn get_entities(&self) -> RwLockReadGuard<'_, Vec<EntityId>> {
        let entities = self.entities.read();
        entities
    }

    pub fn len(&self) -> usize {
        let read_lock = self.entities.read();
        read_lock.len()
    }

    #[inline]
    pub fn add(&self, id: EntityId) -> bool {
        let mut write_lock = self.entities.write();
        let became_active = write_lock.is_empty();
        write_lock.push(id);
        became_active
    }

    #[inline]
    pub fn remove(&self, id: EntityId) -> bool {
        let mut write_lock = self.entities.write();
        if let Some(found) = write_lock.iter().position(|&e| e == id) {
            write_lock.swap_remove(found);
            return write_lock.is_empty();
        }
        false
    }
}
