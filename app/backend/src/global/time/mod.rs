use std::time::{Duration, Instant};

pub struct Time {
    delta: Duration,
    fixed_delta: Duration,
    last_frame: Instant,

    fixed_accumulator: Duration,
    fixed_update_duration: Duration,
}

impl Time {
    pub fn new() -> Self {
        Self {
            delta: Duration::ZERO,
            fixed_delta: Duration::ZERO,
            last_frame: Instant::now(),

            fixed_accumulator: Duration::ZERO,
            fixed_update_duration: Duration::from_millis(50),
        }
    }

    pub fn reset(&mut self) {
        self.delta = self.last_frame.elapsed();
        self.last_frame = Instant::now();
    }

    pub fn delta(&self) -> Duration {
        self.delta
    }

    pub fn fixed_delta(&self) -> Duration {
        self.fixed_delta
    }

    pub fn can_run_fixed_update(&mut self) -> bool {
        if self.fixed_accumulator >= self.fixed_update_duration {
            self.fixed_accumulator = self.delta.clone();

            return true;
        }

        self.fixed_accumulator += self.delta;
        return false;
    }
}
