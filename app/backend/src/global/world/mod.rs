use std::time::{Duration, Instant};

use crate::{
    entities::{entity_id::EntityId, entity_manager::EntityManager},
    spatial_db::grid_manager::GridManager,
};

pub struct World {
    pub entity_manager: EntityManager,
    pub grid_manager: GridManager,
    // pub collisions: CollisionSystem,
}

impl World {
    pub fn new() -> Self {
        Self {
            entity_manager: EntityManager::new(),
            grid_manager: GridManager::new(),
            // collisions: CollisionSystem::new(4),
        }
    }

    pub fn add_entity(
        &mut self,
        position: (f32, f32),
        velocity: (f32, f32),
        facing: Option<(f32, f32)>,
        force: Option<(f32, f32)>,
        mass: f32,
        size: f32,
        sensing_radius: f32,
    ) {
        let id = self.entity_manager.insert(
            position,
            velocity,
            facing,
            force,
            mass,
            size,
            sensing_radius,
        );

        self.grid_manager.insert(id, position);
    }

    pub fn remove_entity(&mut self, id: EntityId) {
        if self.entity_manager.remove(id) {
            self.grid_manager.remove(id);
        }
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

        // let start = Instant::now();
        // self.collisions.detect(&self.grid, &self.entities);
        // println!("[INFO] Detected: {:?}", start.elapsed());
        // duration += start.elapsed();

        /*
            2. Apply forces -> velocities.

            Your SIMD implementation.
        */
        let start = Instant::now();
        self.entity_manager.update_velocities(dt);
        println!("[INFO] Velocities: {:?}", start.elapsed());
        duration += start.elapsed();

        /*
            3. Resolve collisions.

            Several solver iterations improve stability.
        */
        // let start = Instant::now();
        // self.collisions.solve(&mut self.entities);
        // println!("[INFO] Solved: {:?}", start.elapsed());
        // duration += start.elapsed();

        /*
            4. Integrate positions.

            SIMD implementation.
        */
        let start = Instant::now();
        self.entity_manager.update_positions(dt);
        println!("[INFO] Positions: {:?}", start.elapsed());
        duration += start.elapsed();

        /*
            5. Update spatial partition.

            Only entities that crossed a boundary are moved.
        */
        let start = Instant::now();
        self.grid_manager.relocate(
            self.entity_manager.get_ids(),
            self.entity_manager.get_positions_x(),
            self.entity_manager.get_positions_y(),
        );
        println!("[INFO] Relocated: {:?}", start.elapsed());
        duration += start.elapsed();

        println!("[INFO] Total: {:?}", duration);
    }
}
