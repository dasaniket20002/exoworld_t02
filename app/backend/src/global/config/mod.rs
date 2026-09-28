use std::sync::OnceLock;

#[derive(Clone, Copy)]
pub struct Config {
    pub simd_lanes: u8,

    pub max_entities: usize,
    pub world_size: f32,

    pub collision_restitution: f32,
    pub penetration_slop: f32,
    pub correction_percent: f32,

    pub radius_range: (f32, f32),
    pub sensing_radius_range: (f32, f32),
}

static INSTANCE: OnceLock<Config> = OnceLock::new();

impl Config {
    pub fn get_instance() -> &'static Config {
        INSTANCE.get_or_init(|| Config {
            simd_lanes: 16,

            max_entities: 1_000_000,
            world_size: 20_000.0,

            collision_restitution: 1.0,
            correction_percent: 0.8,
            penetration_slop: 0.001,

            radius_range: (0.5, 1.75),
            sensing_radius_range: (0.75, 2.25),
        })
    }
}
