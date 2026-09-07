use std::time::Instant;

use crate::{
    entities::entities_soa::EntitiesSoa,
    global::{config::Config, time::Time},
};

mod entities;
mod global;
mod spatial_db;

fn main() {
    let config = Config::get_instance();
    let mut entities_soa = EntitiesSoa::new(config.max_entities);

    let mut time = Time::new();

    let start = Instant::now();
    println!("[INFO] Spawnning {} entities", config.max_entities);
    (0..config.max_entities).for_each(|_i| {
        let position_x = fastrand::f32() * config.world_size;
        let position_y = fastrand::f32() * config.world_size;
        entities_soa.add(position_x, position_y, 0.0, 0.0, 0.0, 0.0, 1.0);
    });
    println!("[INFO] Spawned in {:?}", start.elapsed());

    loop {
        time.reset();
        let _delta = time.delta();

        if time.can_run_fixed_update() {
            let fixed_delta = time.fixed_delta();

            let start = Instant::now();
            entities_soa.update_velocities(fixed_delta.as_secs_f32());
            entities_soa.update_positions(fixed_delta.as_secs_f32());
            println!("[INFO] Updated in {:?}", start.elapsed());
        }

        // run update here
    }
}
