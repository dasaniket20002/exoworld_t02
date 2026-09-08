#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityId {
    pub index: u32,
    pub generation: u32,
}

impl EntityId {
    #[inline]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }
}
