use std::time::{Duration, Instant};

use crate::{
    collisions::collision_system::CollisionSystem, entities::entity_manager::EntityManager,
    spatial_db::grid_manager::GridManager,
};

pub struct World {}

impl World {
    pub fn add_entity(
        position: (f32, f32),
        velocity: (f32, f32),
        facing: Option<(f32, f32)>,
        force: Option<(f32, f32)>,
        mass: f32,
        size: f32,
        sensing_radius: f32,
    ) {
        let id = {
            let mut entity_manager_write = EntityManager::get_instance().write();
            entity_manager_write.insert(
                position,
                velocity,
                facing,
                force,
                mass,
                size,
                sensing_radius,
            )
        };

        {
            let mut grid_manager_write = GridManager::get_instance().write();
            grid_manager_write.insert(id, position);
        }
    }

    // pub fn remove_entity(id: EntityId) {
    //     let removed = {
    //         let mut entity_manager_write = EntityManager::get_instance().write();
    //         entity_manager_write.remove(id)
    //     };

    //     if removed {
    //         let mut grid_manager_write = GridManager::get_instance().write();
    //         grid_manager_write.remove(id);
    //     }
    // }

    pub fn update(dt: f32) {
        let mut duration = Duration::ZERO;

        /*
            1. Detect current collisions.

            This uses the persistent grid.
        */

        let start = Instant::now();
        {
            let mut collision_system_read = CollisionSystem::get_instance().write();
            collision_system_read.detect_collisions();
        }
        println!("[INFO] Detected: {:?}", start.elapsed());
        duration += start.elapsed();

        /*
            2. Apply forces -> velocities.

            Your SIMD implementation.
        */
        let start = Instant::now();
        {
            let mut entity_manager_write = EntityManager::get_instance().write();
            entity_manager_write.update_velocities(dt);
        }
        println!("[INFO] Velocities: {:?}", start.elapsed());
        duration += start.elapsed();

        /*
            3. Resolve collisions.

            Several solver iterations improve stability.
        */
        let start = Instant::now();
        {
            let collision_system_read = CollisionSystem::get_instance().read();
            collision_system_read.resolve_collisions();
        }
        println!("[INFO] Solved: {:?}", start.elapsed());
        duration += start.elapsed();

        /*
            4. Integrate positions.

            SIMD implementation.
        */
        let start = Instant::now();
        {
            let mut entity_manager_write = EntityManager::get_instance().write();
            entity_manager_write.update_positions(dt);
        }
        println!("[INFO] Positions: {:?}", start.elapsed());
        duration += start.elapsed();

        /*
            5. Update spatial partition.

            Only entities that crossed a boundary are moved.
        */
        let start = Instant::now();
        {
            let mut grid_manager_read = GridManager::get_instance().write();
            grid_manager_read.relocate();
        }
        println!("[INFO] Relocated: {:?}", start.elapsed());
        duration += start.elapsed();

        println!("[INFO] Total: {:?} \n", duration);
    }
}
