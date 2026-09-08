use std::cmp::Reverse;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::ParallelSliceMut;

use crate::entities::entity_id::EntityId;

const INVALID_CELL: u32 = u32::MAX;
const INVALID_LOCATION: u64 = u64::MAX;

#[inline]
fn pack_location(cell: u32, slot: u32) -> u64 {
    ((cell as u64) << 32) | slot as u64
}

#[inline]
fn unpack_location(value: u64) -> (u32, u32) {
    ((value >> 32) as u32, value as u32)
}

#[derive(Default)]
pub struct GridCell {
    pub entities: Vec<EntityId>,
}

#[derive(Clone, Copy, Debug)]
struct Move {
    entity: EntityId,

    // Old grid location.
    from: u32,
    slot: u32,

    // New grid location.
    to: u32,
}

pub struct UniformGrid {
    inv_cell_size: f32,

    pub num_cells_1d: u32,

    cells: Vec<GridCell>,

    /*
        One location per stable EntityId.

        Encoded as:

        upper 32 bits = cell
        lower 32 bits = slot inside cell
    */
    locations: Vec<AtomicU64>,

    /*
        Scratch destination cell for every stable entity.

        This is atomic only because the destination-detection pass
        executes in parallel.
    */
    next_cell: Vec<AtomicU32>,

    /*
        Reused scratch array.

        After prefix sum:

        ranges[cell]     = first mover belonging to cell
        ranges[cell + 1] = one-past-last mover
    */
    ranges: Vec<usize>,
}

impl UniformGrid {
    pub fn new(world_size: f32, cell_size: f32, max_entities: usize) -> Self {
        assert!(world_size > 0.0);
        assert!(cell_size > 0.0);

        let inv_cell_size = 1.0 / cell_size;
        let num_cells_1d = (world_size * inv_cell_size).ceil() as u32;

        let cell_count = (num_cells_1d * num_cells_1d) as usize;

        let cells = (0..cell_count).map(|_| GridCell::default()).collect();

        let locations = (0..max_entities)
            .map(|_| AtomicU64::new(INVALID_LOCATION))
            .collect();

        let next_cell = (0..max_entities)
            .map(|_| AtomicU32::new(INVALID_CELL))
            .collect();

        Self {
            inv_cell_size,

            num_cells_1d,

            cells,

            locations,
            next_cell,

            ranges: Vec::with_capacity(cell_count + 1),
        }
    }

    #[inline]
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    #[inline]
    pub fn cell_index(&self, cell_x: u32, cell_y: u32) -> u32 {
        debug_assert!(cell_x < self.num_cells_1d);
        debug_assert!(cell_y < self.num_cells_1d);

        cell_y * self.num_cells_1d + cell_x
    }

    #[inline]
    fn cell_from_position(&self, x: f32, y: f32) -> u32 {
        let mut cell_x = (x * self.inv_cell_size).floor() as i32;
        let mut cell_y = (y * self.inv_cell_size).floor() as i32;

        cell_x = cell_x.clamp(0, self.num_cells_1d as i32 - 1);
        cell_y = cell_y.clamp(0, self.num_cells_1d as i32 - 1);

        self.cell_index(cell_x as u32, cell_y as u32)
    }

    #[inline]
    pub fn entities_in_cell(&self, cell_x: u32, cell_y: u32) -> &[EntityId] {
        let cell = self.cell_index(cell_x, cell_y);
        &self.cells[cell as usize].entities
    }

    #[inline]
    pub fn entity_location(&self, entity: EntityId) -> Option<(u32, u32)> {
        let value = self.locations[entity.index as usize].load(Ordering::Relaxed);

        if value == INVALID_LOCATION {
            None
        } else {
            Some(unpack_location(value))
        }
    }

    pub fn insert(&mut self, entity: EntityId, x: f32, y: f32) {
        let cell = self.cell_from_position(x, y);

        let entities = &mut self.cells[cell as usize].entities;

        let slot = entities.len() as u32;

        entities.push(entity);

        self.locations[entity.index as usize].store(pack_location(cell, slot), Ordering::Relaxed);

        self.next_cell[entity.index as usize].store(cell, Ordering::Relaxed);
    }

    pub fn remove(&mut self, entity: EntityId) -> bool {
        let location = self.locations[entity.index as usize].load(Ordering::Relaxed);

        if location == INVALID_LOCATION {
            return false;
        }

        let (cell, slot) = unpack_location(location);

        let entities = &mut self.cells[cell as usize].entities;

        let slot = slot as usize;

        debug_assert!(slot < entities.len());
        debug_assert_eq!(entities[slot], entity);

        let last_entity = entities.pop().unwrap();

        if slot < entities.len() {
            entities[slot] = last_entity;

            self.locations[last_entity.index as usize]
                .store(pack_location(cell, slot as u32), Ordering::Relaxed);
        }

        self.locations[entity.index as usize].store(INVALID_LOCATION, Ordering::Relaxed);

        self.next_cell[entity.index as usize].store(INVALID_CELL, Ordering::Relaxed);

        true
    }

    /*
        Builds:

            ranges[i] = count of movers whose key == i

        then converts counts into prefix sums.

        After this:

            ranges[i]..ranges[i + 1]

        is the range belonging to cell i.
    */
    fn build_ranges_from_source(&mut self, movers: &Vec<Move>) {
        self.ranges.fill(0);

        for movement in movers {
            self.ranges[movement.from as usize + 1] += 1;
        }

        for i in 1..self.ranges.len() {
            let previous = self.ranges[i - 1];
            self.ranges[i] += previous;
        }
    }

    fn build_ranges_from_destination(&mut self, movers: &Vec<Move>) {
        self.ranges.fill(0);

        for movement in movers {
            self.ranges[movement.to as usize + 1] += 1;
        }

        for i in 1..self.ranges.len() {
            let previous = self.ranges[i - 1];
            self.ranges[i] += previous;
        }
    }

    fn detect_moved_entities(
        &self,
        entity_ids: &[EntityId],
        position_x: &[f32],
        position_y: &[f32],
    ) -> Vec<Move> {
        let locations = &self.locations;
        let next_cell = &self.next_cell;

        let inv_cell_size = self.inv_cell_size;

        let num_cells_1d = self.num_cells_1d;

        let movers = entity_ids
            .par_iter()
            .enumerate()
            .filter_map(|(dense_index, &entity)| {
                let location = locations[entity.index as usize].load(Ordering::Relaxed);

                debug_assert_ne!(location, INVALID_LOCATION);

                if location == INVALID_LOCATION {
                    return None;
                }

                let (from, slot) = unpack_location(location);

                let mut cell_x = (position_x[dense_index] * inv_cell_size).floor() as i32;
                let mut cell_y = (position_y[dense_index] * inv_cell_size).floor() as i32;

                cell_x = cell_x.clamp(0, num_cells_1d as i32 - 1);
                cell_y = cell_y.clamp(0, num_cells_1d as i32 - 1);

                let to = (cell_y as u32) * num_cells_1d + cell_x as u32;

                next_cell[entity.index as usize].store(to, Ordering::Relaxed);

                if from == to {
                    None
                } else {
                    Some(Move {
                        entity,
                        from,
                        slot,
                        to,
                    })
                }
            })
            .collect::<Vec<_>>();

        movers
    }

    fn remove_movers_parallel(&mut self, movers: &mut Vec<Move>) {
        /*
            ---------------------------------------------------------
            Sort movers by source cell and descending source slot.

            Descending slot is important.

            If a cell contains:

                [A B C D E F]

            and B and E leave:

                remove slot 4
                remove slot 1

            so earlier removals never invalidate a lower slot.
            ---------------------------------------------------------
        */

        movers.par_sort_unstable_by_key(|movement| (movement.from, Reverse(movement.slot)));

        self.build_ranges_from_source(&movers);

        self.cells
            .par_iter_mut()
            .enumerate()
            .for_each(|(cell_index, cell)| {
                let start = self.ranges[cell_index];

                let end = self.ranges[cell_index + 1];

                if start == end {
                    return;
                }

                let cell_index = cell_index as u32;

                for movement in &movers[start..end] {
                    let slot = movement.slot as usize;

                    debug_assert_eq!(movement.from, cell_index);
                    debug_assert!(slot < cell.entities.len());
                    debug_assert_eq!(cell.entities[slot], movement.entity);

                    let last_entity = cell.entities.pop().expect("grid cell unexpectedly empty");

                    if slot < cell.entities.len() {
                        cell.entities[slot] = last_entity;

                        self.locations[last_entity.index as usize]
                            .store(pack_location(cell_index, slot as u32), Ordering::Relaxed);
                    }

                    self.locations[movement.entity.index as usize]
                        .store(INVALID_LOCATION, Ordering::Relaxed);
                }
            });
    }

    fn append_movers_parallel(&mut self, movers: &mut Vec<Move>) {
        /*
            ---------------------------------------------------------
            PHASE 4
            Sort movers by destination cell.
            ---------------------------------------------------------
        */

        movers.par_sort_unstable_by_key(|movement| movement.to);
        self.build_ranges_from_destination(&movers);

        self.cells
            .par_iter_mut()
            .enumerate()
            .for_each(|(cell_index, cell)| {
                let start = self.ranges[cell_index];
                let end = self.ranges[cell_index + 1];

                if start == end {
                    return;
                }

                let destination = cell_index as u32;
                let incoming = end - start;

                cell.entities.reserve(incoming);

                let base_slot = cell.entities.len();

                for (offset, movement) in movers[start..end].iter().enumerate() {
                    debug_assert_eq!(movement.to, destination);

                    let slot = base_slot + offset;
                    cell.entities.push(movement.entity);
                    self.locations[movement.entity.index as usize]
                        .store(pack_location(destination, slot as u32), Ordering::Relaxed);
                }
            });
    }

    /*
        Relocates all entities whose position crossed a cell boundary.

        This does NOT rebuild the grid.

        It only removes movers from their current cells and appends
        them to their destination cells.
    */
    pub fn relocate(&mut self, entity_ids: &[EntityId], position_x: &[f32], position_y: &[f32]) {
        debug_assert_eq!(entity_ids.len(), position_x.len());

        debug_assert_eq!(entity_ids.len(), position_y.len());

        /*
            ---------------------------------------------------------
            PHASE 1
            Detect entities that changed cells.
            ---------------------------------------------------------
        */

        let mut movers = self.detect_moved_entities(entity_ids, position_x, position_y);

        if movers.is_empty() {
            return;
        }

        /*
            ---------------------------------------------------------
            PHASE 2
            Remove movers from source cells in parallel.

            Each GridCell is exclusively owned by one Rayon worker
            during this operation.
            ---------------------------------------------------------
        */

        self.remove_movers_parallel(&mut movers);

        /*
            ---------------------------------------------------------
            PHASE 3
            Append movers to destination cells in parallel.

            Each destination cell owns exactly one range in movers,
            so there is no mutex and no atomic push.
            ---------------------------------------------------------
        */

        self.append_movers_parallel(&mut movers);
    }
}
