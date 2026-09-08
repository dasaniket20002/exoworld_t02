use std::time::Instant;

use crate::{
    entities::{entities_soa::EntitiesSoa, entity_id::EntityId},
    global::config::Config,
    spatial_db::uniform_grid::UniformGrid,
};

pub struct World {
    pub entities: EntitiesSoa,
    pub grid: UniformGrid,
}

impl World {
    pub fn new() -> Self {
        let cfg = Config::get_instance();
        Self {
            entities: EntitiesSoa::new(cfg.max_entities),
            grid: UniformGrid::new(cfg.world_size, cfg.cell_size, cfg.max_entities),
        }
    }

    pub fn add_entity(
        &mut self,
        position_x: f32,
        position_y: f32,
        velocity_x: f32,
        velocity_y: f32,
        force_x: f32,
        force_y: f32,
        mass: f32,
    ) -> EntityId {
        let entity = self.entities.add(
            position_x, position_y, velocity_x, velocity_y, force_x, force_y, mass,
        );

        self.grid.insert(entity, position_x, position_y);

        entity
    }

    pub fn remove_entity(&mut self, entity: EntityId) -> bool {
        if !self.entities.contains(entity) {
            return false;
        }

        /*
            Remove from the spatial structure first.
        */
        self.grid.remove(entity);

        /*
            Then compact the SoA.
        */
        self.entities.remove(entity)
    }

    pub fn update(&mut self, dt: f32) {
        /*
            1. SIMD physics integration.
        */
        let start = Instant::now();
        self.entities.update_velocities(dt);
        self.entities.update_positions(dt);
        println!("[INFO] Updated in {:?}", start.elapsed());

        /*
            2. Relocate entities that crossed a cell boundary.
        */
        let start = Instant::now();
        self.grid.relocate(
            self.entities.entity_ids(),
            self.entities.position_x(),
            self.entities.position_y(),
        );
        println!("[INFO] Relocated in {:?}", start.elapsed());
    }

    #[inline]
    pub fn entity_position(&self, entity: EntityId) -> Option<(f32, f32)> {
        let dense = self.entities.dense_index(entity)?;

        Some((
            self.entities.position_x()[dense],
            self.entities.position_y()[dense],
        ))
    }
}
