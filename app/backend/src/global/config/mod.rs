use std::sync::OnceLock;

#[derive(Clone, Copy)]
pub struct Config {
    pub max_entities: usize,
    pub world_size: f32,
}

static INSTANCE: OnceLock<Config> = OnceLock::new();

impl Config {
    pub fn get_instance() -> &'static Config {
        INSTANCE.get_or_init(|| Config {
            max_entities: 1_000_000,
            world_size: 20_000.0,
        })
    }
}
