use std::time::Instant;

use crate::global::{config::Config, time::Time, world::World};

mod collisions;
mod entities;
mod global;
mod spatial_db;

fn main() {
    let config = Config::get_instance();
    let mut time = Time::new();

    let start = Instant::now();
    println!("[INFO] Spawnning {} entities", config.max_entities);
    (0..config.max_entities).for_each(|_i| {
        let position_x = fastrand::f32() * config.world_size;
        let position_y = fastrand::f32() * config.world_size;

        let velocity_x = fastrand::f32();
        let velocity_y = fastrand::f32();

        let facing_x = fastrand::f32();
        let facing_y = fastrand::f32();

        let sensing_radius = fastrand::f32() * (config.radius_range.1 /* max */ - config.radius_range.0 /* min */) + config.radius_range.0 /* min */;
        let radius = fastrand::f32() * (config.sensing_radius_range.1 /* max */ - sensing_radius /* min */) + sensing_radius /* min */;

        World::add_entity(
            (position_x, position_y),
            (velocity_x, velocity_y),
            Some((facing_x, facing_y)),
            None,
            0.0,
            radius,
            sensing_radius,
        );
    });
    println!("[INFO] Spawned in {:?}", start.elapsed());

    loop {
        time.reset();
        let _delta = time.delta();

        if time.can_run_fixed_update() {
            let fixed_delta = time.fixed_delta();

            World::update(fixed_delta.as_secs_f32());
        }

        // run update here
    }
}
