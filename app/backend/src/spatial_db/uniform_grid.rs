use std::sync::atomic::{AtomicU64, Ordering};

use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{entities::entity_id::EntityId, global::config::Config};

const INVALID_LOCATION: u64 = u64::MAX;

#[inline]
fn pack_location(cell: u32, slot: u32) -> u64 {
    ((cell as u64) << 32) | slot as u64
}

#[inline]
fn unpack_location(location: u64) -> (u32, u32) {
    ((location >> 32) as u32, location as u32)
}

pub struct GridCell {
    pub entities: Vec<u32>,
}

impl GridCell {
    #[inline]
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
        }
    }
}

pub struct UniformGrid {
    pub world_size: f32,

    pub cell_size: f32,
    inv_cell_size: f32,

    pub num_cells_axis: u32,

    /*
        Persistent cells.

        Each entry is a DENSE SOA INDEX.
    */
    cells: Vec<GridCell>,

    /*
        Stable EntityId -> (cell, slot)

        Packed:

            upper 32 bits = cell
            lower 32 bits = slot
    */
    locations: Vec<AtomicU64>,

    /*
        Per-dense-index destination cell.

        Reused every frame.
    */
    next_cell: Vec<u32>,

    /*
        Reused movement buffers.
    */
    movers: Vec<u32>,
    ordered_movers: Vec<u32>,

    /*
        Counting-sort style scratch.

        destination_offsets[cell]
        .. destination_offsets[cell + 1]

        is the range occupied by entities
        moving into that destination cell.
    */
    destination_offsets: Vec<usize>,
    destination_cursors: Vec<usize>,
}

impl UniformGrid {
    pub fn new() -> Self {
        let config = Config::get_instance();

        assert!(config.world_size > 0.0);
        assert!(config.cell_size > 0.0);

        let inv_cell_size = 1.0 / config.cell_size;

        let num_cells_axis = (config.world_size * inv_cell_size).ceil() as u32;

        let cell_count = num_cells_axis as usize * num_cells_axis as usize;

        let cells = (0..cell_count).map(|_| GridCell::new()).collect();

        let locations = (0..config.max_entities)
            .map(|_| AtomicU64::new(INVALID_LOCATION))
            .collect();

        Self {
            world_size: config.world_size,

            cell_size: config.cell_size,
            inv_cell_size,

            num_cells_axis,

            cells,

            locations,

            next_cell: vec![0; config.max_entities],

            movers: Vec::with_capacity(config.max_entities),

            ordered_movers: Vec::with_capacity(config.max_entities),

            destination_offsets: vec![0; cell_count + 1],

            destination_cursors: vec![0; cell_count],
        }
    }

    #[inline]
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    #[inline]
    pub fn cells(&self) -> &[GridCell] {
        &self.cells
    }

    #[inline]
    pub fn cell_entities(&self, cell: usize) -> &[u32] {
        &self.cells[cell].entities
    }

    #[inline(always)]
    pub fn cell_index(&self, x: u32, y: u32) -> u32 {
        y * self.num_cells_axis + x
    }

    #[inline(always)]
    fn position_to_cell(&self, x: f32, y: f32) -> u32 {
        let cx = (x * self.inv_cell_size)
            .floor()
            .clamp(0.0, self.num_cells_axis as f32 - 1.0) as u32;

        let cy = (y * self.inv_cell_size)
            .floor()
            .clamp(0.0, self.num_cells_axis as f32 - 1.0) as u32;

        self.cell_index(cx, cy)
    }

    pub fn insert(&mut self, entity: EntityId, dense_index: u32, x: f32, y: f32) {
        let cell = self.position_to_cell(x, y);

        let entities = &mut self.cells[cell as usize].entities;

        let slot = entities.len() as u32;

        entities.push(dense_index);

        self.locations[entity.index as usize].store(pack_location(cell, slot), Ordering::Relaxed);
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

        let moved_dense = entities.pop().unwrap();

        if slot < entities.len() {
            entities[slot] = moved_dense;

            /*
                Find the stable ID of the entity that was
                moved into this grid slot.

                The caller owns the dense SoA, so this method
                needs the dense -> EntityId table in a more
                sophisticated implementation.

                See `remove_with_entities()` below.
            */
        }

        self.locations[entity.index as usize].store(INVALID_LOCATION, Ordering::Relaxed);

        true
    }

    pub fn remove_with_entities(&mut self, entity: EntityId, entity_ids: &[EntityId]) -> bool {
        let location = self.locations[entity.index as usize].load(Ordering::Relaxed);

        if location == INVALID_LOCATION {
            return false;
        }

        let (cell, slot) = unpack_location(location);

        let entities = &mut self.cells[cell as usize].entities;

        let slot = slot as usize;

        debug_assert!(slot < entities.len());

        debug_assert!(entities[slot] < entity_ids.len() as u32);

        let last_dense = entities.pop().unwrap();

        if slot < entities.len() {
            entities[slot] = last_dense;

            let moved_entity = entity_ids[last_dense as usize];

            self.locations[moved_entity.index as usize]
                .store(pack_location(cell, slot as u32), Ordering::Relaxed);
        }

        self.locations[entity.index as usize].store(INVALID_LOCATION, Ordering::Relaxed);

        true
    }

    pub fn update_dense_index(&mut self, entity: EntityId, new_dense_index: u32) {
        let location = self.locations[entity.index as usize].load(Ordering::Relaxed);

        debug_assert_ne!(location, INVALID_LOCATION);

        let (cell, slot) = unpack_location(location);

        self.cells[cell as usize].entities[slot as usize] = new_dense_index;
    }

    /*
        ------------------------------------------------------------
        RELOCATION
        ------------------------------------------------------------

        No sorting.

        1. Determine destination cell for every dense entity.
        2. Compact each current cell in parallel.
        3. Collect movers.
        4. Count destination cells.
        5. Prefix sum.
        6. Scatter movers into destination ranges.
        7. Insert destination ranges in parallel.
    */
    pub fn relocate(&mut self, entity_ids: &[EntityId], position_x: &[f32], position_y: &[f32]) {
        let entity_count = entity_ids.len();

        if entity_count == 0 {
            return;
        }

        /*
            --------------------------------------------------------
            1. Destination cell for every entity.
            --------------------------------------------------------
        */

        let inv_cell_size = self.inv_cell_size;
        let num_cells_axis = self.num_cells_axis;

        self.next_cell[..entity_count]
            .par_iter_mut()
            .zip(position_x[..entity_count].par_iter())
            .zip(position_y[..entity_count].par_iter())
            .for_each(|((destination, &x), &y)| {
                let cx = (x * inv_cell_size)
                    .floor()
                    .clamp(0.0, num_cells_axis as f32 - 1.0) as u32;

                let cy = (y * inv_cell_size)
                    .floor()
                    .clamp(0.0, num_cells_axis as f32 - 1.0) as u32;

                *destination = cy * num_cells_axis + cx;
            });

        /*
            --------------------------------------------------------
            2. Remove movers from old cells.

            Each cell is exclusively owned by one Rayon worker.

            Staying entities are compacted in-place.

            No swap_remove is used here because we're removing
            potentially many entities from one cell.
            --------------------------------------------------------
        */

        let next_cell = &self.next_cell;

        let locations = &self.locations;

        self.movers = self
            .cells
            .par_iter_mut()
            .enumerate()
            .fold(
                || Vec::<u32>::with_capacity(16),
                |mut local, (cell_index, cell)| {
                    let mut write = 0usize;

                    let old_len = cell.entities.len();

                    for read in 0..old_len {
                        let dense = cell.entities[read];

                        let destination = next_cell[dense as usize];

                        if destination == cell_index as u32 {
                            /*
                                Entity stays in this cell.

                                Compact if necessary.
                            */
                            if write != read {
                                cell.entities[write] = dense;

                                let entity = entity_ids[dense as usize];

                                locations[entity.index as usize].store(
                                    pack_location(cell_index as u32, write as u32),
                                    Ordering::Relaxed,
                                );
                            }

                            write += 1;
                        } else {
                            /*
                                Entity leaves this cell.
                            */
                            local.push(dense);
                        }
                    }

                    cell.entities.truncate(write);

                    local
                },
            )
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            });

        if self.movers.is_empty() {
            return;
        }

        /*
            --------------------------------------------------------
            3. Count movers per destination cell.

            This replaces:

                par_sort_unstable_by_key()

            with a linear pass.
            --------------------------------------------------------
        */

        self.destination_offsets.fill(0);

        for &dense in &self.movers {
            let destination = self.next_cell[dense as usize] as usize;

            self.destination_offsets[destination + 1] += 1;
        }

        /*
            --------------------------------------------------------
            4. Prefix sum.

            After this:

                offsets[cell] ..
                offsets[cell + 1]

            is the destination range.
            --------------------------------------------------------
        */

        for cell in 1..self.destination_offsets.len() {
            let previous = self.destination_offsets[cell - 1];

            self.destination_offsets[cell] += previous;
        }

        /*
            --------------------------------------------------------
            5. Reusable cursors.
            --------------------------------------------------------
        */

        self.destination_cursors
            .copy_from_slice(&self.destination_offsets[..self.cells.len()]);

        /*
            --------------------------------------------------------
            6. Scatter movers into destination ranges.

            Still O(M).

            No comparison sort.
            --------------------------------------------------------
        */

        self.ordered_movers.resize(self.movers.len(), 0);

        for &dense in &self.movers {
            let destination = self.next_cell[dense as usize] as usize;

            let slot = self.destination_cursors[destination];

            self.ordered_movers[slot] = dense;

            self.destination_cursors[destination] += 1;
        }

        /*
            --------------------------------------------------------
            7. Insert destination ranges in parallel.

            Each destination cell is owned by exactly one worker.
            --------------------------------------------------------
        */

        let ordered = &self.ordered_movers;

        let offsets = &self.destination_offsets;

        self.cells
            .par_iter_mut()
            .enumerate()
            .for_each(|(cell_index, cell)| {
                let start = offsets[cell_index];

                let end = offsets[cell_index + 1];

                if start == end {
                    return;
                }

                let incoming = end - start;

                cell.entities.reserve(incoming);

                let base_slot = cell.entities.len();

                for offset in 0..incoming {
                    let dense = ordered[start + offset];

                    let slot = base_slot + offset;

                    cell.entities.push(dense);

                    let entity = entity_ids[dense as usize];

                    locations[entity.index as usize].store(
                        pack_location(cell_index as u32, slot as u32),
                        Ordering::Relaxed,
                    );
                }
            });
    }
}
