use std::time::Instant;

use crate::global::{config::Config, time::Time, world::World};

mod entities;
mod global;
mod spatial_db;

fn main() {
    let config = Config::get_instance();
    let mut world = World::new();
    let mut time = Time::new();

    let start = Instant::now();
    println!("[INFO] Spawnning {} entities", config.max_entities);
    (0..config.max_entities).for_each(|_i| {
        let position_x = fastrand::f32() * config.world_size;
        let position_y = fastrand::f32() * config.world_size;

        let velocity_x = fastrand::f32();
        let velocity_y = fastrand::f32();

        let radius = fastrand::f32() * (2.25 - 0.75) + 0.75;

        world.add_entity(
            position_x, position_y, velocity_x, velocity_y, 0.0, 0.0, 1.0, radius,
        );
    });
    println!("[INFO] Spawned in {:?}", start.elapsed());

    loop {
        time.reset();
        let _delta = time.delta();

        if time.can_run_fixed_update() {
            let fixed_delta = time.fixed_delta();

            world.update(fixed_delta.as_secs_f32());
        }

        // run update here
    }
}
