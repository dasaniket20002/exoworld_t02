use std::time::{Duration, Instant};

use crate::{
    entities::{entities_soa::EntitiesSoa, entity_id::EntityId},
    spatial_db::{collision::CollisionSystem, uniform_grid::UniformGrid},
};

pub struct World {
    pub entities: EntitiesSoa,
    pub grid: UniformGrid,
    pub collisions: CollisionSystem,
}

impl World {
    pub fn new() -> Self {
        Self {
            entities: EntitiesSoa::new(),

            grid: UniformGrid::new(),

            collisions: CollisionSystem::new(4),
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
        radius: f32,
    ) -> EntityId {
        let (entity, dense_index) = self.entities.add(
            position_x, position_y, velocity_x, velocity_y, force_x, force_y, mass, radius,
        );

        self.grid
            .insert(entity, dense_index, position_x, position_y);

        entity
    }

    pub fn remove_entity(&mut self, entity: EntityId) -> bool {
        if !self.entities.contains(entity) {
            return false;
        }

        self.grid.remove(entity);

        let moved = self.entities.remove(entity);

        if let Some((moved_entity, new_dense_index)) = moved {
            self.grid.update_dense_index(moved_entity, new_dense_index);
        }

        true
    }

    pub fn update(&mut self, dt: f32) {
        /*
            The grid represents the positions at the start
            of this frame.
        */

        /*
            1. Detect current collisions.

            This uses the persistent grid.
        */
        let mut duration = Duration::ZERO;

        let start = Instant::now();
        self.collisions.detect(&self.grid, &self.entities);
        println!("[INFO] Detected: {:?}", start.elapsed());
        duration += start.elapsed();

        /*
            2. Apply forces -> velocities.

            Your SIMD implementation.
        */
        let start = Instant::now();
        self.entities.update_velocities(dt);
        println!("[INFO] Velocities: {:?}", start.elapsed());
        duration += start.elapsed();

        /*
            3. Resolve collisions.

            Several solver iterations improve stability.
        */
        let start = Instant::now();
        self.collisions.solve(&mut self.entities);
        println!("[INFO] Solved: {:?}", start.elapsed());
        duration += start.elapsed();

        /*
            4. Integrate positions.

            SIMD implementation.
        */
        let start = Instant::now();
        self.entities.update_positions(dt);
        println!("[INFO] Positions: {:?}", start.elapsed());
        duration += start.elapsed();

        /*
            5. Update spatial partition.

            Only entities that crossed a boundary are moved.
        */
        let start = Instant::now();
        self.grid.relocate(
            &self.entities.entity_ids,
            &self.entities.position_x,
            &self.entities.position_y,
        );
        println!("[INFO] Relocated: {:?}", start.elapsed());
        duration += start.elapsed();

        println!("[INFO] Total: {:?}", duration);
    }
}
