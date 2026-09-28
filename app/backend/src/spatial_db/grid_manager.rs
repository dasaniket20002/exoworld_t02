use std::sync::OnceLock;

use parking_lot::RwLock;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    entities::{entity_id::EntityId, entity_manager::EntityManager},
    global::config::Config,
    spatial_db::grid_cell::GridCell,
};

const INVALID_CELL: usize = usize::MAX;

static INSTANCE: OnceLock<RwLock<GridManager>> = OnceLock::new();

#[derive(Clone, Copy)]
struct EntityMove {
    id: EntityId,
    old_cell: usize,
    new_cell: usize,
}

pub struct GridManager {
    cell_size: f32,
    world_size: f32,
    cell_count_axis: usize,
    cells: Vec<GridCell>,
    id_to_cell: Vec<usize>,
    // Dense list of currently non-empty cells.
    active_cells: Vec<usize>,
    // cell index -> position inside active_cells.
    // INVALID_CELL means inactive.
    active_cell_index: Vec<usize>,
}

impl GridManager {
    pub fn get_instance() -> &'static RwLock<GridManager> {
        INSTANCE.get_or_init(|| RwLock::new(GridManager::new()))
    }

    fn new() -> Self {
        let config = Config::get_instance();
        let cell_size = 2.0 * config.radius_range.1;

        let cell_count_axis = (config.world_size / cell_size).ceil() as usize;
        let total_cells = cell_count_axis * cell_count_axis;

        let cells = (0..total_cells).map(|_| GridCell::new()).collect();
        let id_to_cell = vec![INVALID_CELL; config.max_entities];

        let active_cell_index = vec![INVALID_CELL; total_cells];

        Self {
            cells,
            cell_count_axis,
            cell_size,
            world_size: config.world_size,
            id_to_cell,
            active_cells: Vec::new(),
            active_cell_index,
        }
    }

    #[inline]
    pub fn get_cell_count_axis(&self) -> usize {
        self.cell_count_axis
    }

    #[inline]
    pub fn get_cells(&self) -> &[GridCell] {
        &self.cells
    }

    #[inline]
    pub fn get_active_cells(&self) -> &[usize] {
        &self.active_cells
    }

    #[inline]
    pub fn is_active_cell(&self, cell_index: usize) -> bool {
        self.active_cell_index[cell_index] != INVALID_CELL
    }

    #[inline]
    pub fn get_cell_index(&self, position: (f32, f32)) -> usize {
        let pos_x = position.0.clamp(0.0, self.world_size);
        let pos_y = position.1.clamp(0.0, self.world_size);

        let cell_x = ((pos_x / self.cell_size).floor() as usize).min(self.cell_count_axis - 1);
        let cell_y = ((pos_y / self.cell_size).floor() as usize).min(self.cell_count_axis - 1);

        cell_y * self.cell_count_axis + cell_x
    }

    #[inline]
    fn activate_cell(&mut self, cell_index: usize) {
        if self.active_cell_index[cell_index] != INVALID_CELL {
            return;
        }

        let active_index = self.active_cells.len();
        self.active_cells.push(cell_index);
        self.active_cell_index[cell_index] = active_index;
    }

    #[inline]
    fn deactivate_cell(&mut self, cell_index: usize) {
        let active_index = self.active_cell_index[cell_index];

        if active_index == INVALID_CELL {
            return;
        }

        let last_cell = self.active_cells.pop().unwrap();

        if last_cell != cell_index {
            self.active_cells[active_index] = last_cell;
            self.active_cell_index[last_cell] = active_index;
        }

        self.active_cell_index[cell_index] = INVALID_CELL;
    }

    fn refresh_active_cells(&mut self, dirty_cells: &mut Vec<usize>) {
        dirty_cells.sort_unstable();
        dirty_cells.dedup();

        for &cell_index in dirty_cells.iter() {
            let is_empty = self.cells[cell_index].len() == 0;

            let active = self.active_cell_index[cell_index] != INVALID_CELL;

            match (active, is_empty) {
                // Active cell still contains entities.
                (true, false) => {}

                // Inactive -> active.
                (false, false) => {
                    self.activate_cell(cell_index);
                }

                // Active -> inactive.
                (true, true) => {
                    self.deactivate_cell(cell_index);
                }

                // Still inactive.
                (false, true) => {}
            }
        }
    }

    #[inline]
    fn find_relocation(&self, id: EntityId, position: (f32, f32)) -> Option<EntityMove> {
        let new_cell_idx = self.get_cell_index(position);
        let old_cell_idx = self.id_to_cell[id.index as usize];

        if old_cell_idx == INVALID_CELL || old_cell_idx == new_cell_idx {
            return None;
        }

        Some(EntityMove {
            id,
            old_cell: old_cell_idx,
            new_cell: new_cell_idx,
        })
    }

    #[inline]
    fn membership_changes(&self, moves: &Vec<EntityMove>) {
        moves.par_iter().for_each(|movement| {
            self.cells[movement.old_cell].remove(movement.id);
            self.cells[movement.new_cell].add(movement.id);
        });
    }

    #[inline]
    fn collect_dirty_cells(&mut self, moves: &Vec<EntityMove>) -> Vec<usize> {
        let mut dirty_cells = Vec::with_capacity(moves.len() * 2);

        for movement in moves {
            self.id_to_cell[movement.id.index as usize] = movement.new_cell;

            dirty_cells.push(movement.old_cell);
            dirty_cells.push(movement.new_cell);
        }

        dirty_cells
    }

    pub fn relocate(&mut self) {
        let entity_manager_read = EntityManager::get_instance().read();

        let ids = entity_manager_read.get_ids();
        let positions_x = entity_manager_read.get_positions_x();
        let positions_y = entity_manager_read.get_positions_y();

        // Phase 1: determine all entity movements.
        let moves = ids
            .par_iter()
            .zip(positions_x.par_iter().zip(positions_y.par_iter()))
            .filter_map(|(&id, (&pos_x, &pos_y))| self.find_relocation(id, (pos_x, pos_y)))
            .collect::<Vec<_>>();

        if moves.is_empty() {
            return;
        }

        self.membership_changes(&moves);

        let mut dirty_cells = self.collect_dirty_cells(&moves);

        self.refresh_active_cells(&mut dirty_cells);
    }

    pub fn insert(&mut self, id: EntityId, position: (f32, f32)) {
        let new_cell_idx = self.get_cell_index(position);

        let new_cell = &self.cells[new_cell_idx];
        let became_active = new_cell.add(id);

        self.id_to_cell[id.index as usize] = new_cell_idx;

        if became_active {
            self.activate_cell(new_cell_idx);
        }
    }

    // pub fn remove(&mut self, id: EntityId) {
    //     let old_cell_idx = self.id_to_cell[id.index as usize];

    //     if old_cell_idx == INVALID_CELL {
    //         return;
    //     }

    //     let old_cell = &self.cells[old_cell_idx];
    //     let became_empty = old_cell.remove(id);

    //     self.id_to_cell[id.index as usize] = INVALID_CELL;

    //     if became_empty {
    //         self.deactivate_cell(old_cell_idx);
    //     }
    // }
}
