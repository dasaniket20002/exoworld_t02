use std::sync::OnceLock;

#[derive(Clone, Copy)]
pub struct Config {
    pub simd_lanes: u8,
    pub max_entities: usize,
    pub world_size: f32,
    pub cell_size: f32,
    pub collision_restitution: f32,
}

static INSTANCE: OnceLock<Config> = OnceLock::new();

impl Config {
    pub fn get_instance() -> &'static Config {
        INSTANCE.get_or_init(|| Config {
            simd_lanes: 16,
            max_entities: 1_000_000,
            world_size: 20_000.0,
            cell_size: 32.0,
            collision_restitution: 1.0,
        })
    }
}
