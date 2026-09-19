use parking_lot::RwLock;

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

    pub fn for_each(&self, mut f: impl FnMut(EntityId)) {
        let entities = self.entities.read();

        for &id in entities.iter() {
            f(id);
        }
    }

    pub fn len(&self) -> usize {
        let read_lock = self.entities.read();
        read_lock.len()
    }

    #[inline]
    pub fn add(&self, id: EntityId) {
        let mut write_lock = self.entities.write();
        write_lock.push(id);
    }

    #[inline]
    pub fn remove(&self, id: EntityId) {
        let mut write_lock = self.entities.write();
        if let Some(found) = write_lock.iter().position(|&e| e == id) {
            write_lock.swap_remove(found);
        }
    }
}
