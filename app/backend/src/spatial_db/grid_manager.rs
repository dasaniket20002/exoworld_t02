use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    entities::entity_id::EntityId, global::config::Config, spatial_db::grid_cell::GridCell,
};

const INVALID_CELL: usize = usize::MAX;

pub struct GridManager {
    cell_size: f32,
    world_size: f32,
    cell_count_axis: usize,
    cells: Vec<GridCell>,
    id_to_cell: Vec<AtomicUsize>,
}

impl GridManager {
    pub fn new() -> Self {
        let config = Config::get_instance();
        let cell_size = 2.0 * config.sensing_radius_range.1;

        let cell_count_axis = (config.world_size / cell_size).ceil() as usize;
        let total_cells = cell_count_axis * cell_count_axis;

        let cells = (0..total_cells).map(|_| GridCell::new()).collect();
        let id_to_cell = (0..config.max_entities)
            .map(|_| AtomicUsize::new(INVALID_CELL))
            .collect();

        Self {
            cells,
            cell_count_axis,
            cell_size,
            world_size: config.world_size,
            id_to_cell,
        }
    }

    #[inline]
    pub fn get_cell_index(&self, position: (f32, f32)) -> usize {
        let pos_x = position.0.clamp(0.0, self.world_size);
        let pos_y = position.1.clamp(0.0, self.world_size);

        let cell_x = ((pos_x / self.cell_size).floor() as usize).min(self.cell_count_axis - 1);
        let cell_y = ((pos_y / self.cell_size).floor() as usize).min(self.cell_count_axis - 1);

        cell_y * self.cell_count_axis + cell_x
    }

    fn _relocate(&self, id: EntityId, position: (f32, f32)) {
        let new_cell_idx = self.get_cell_index(position);
        let old_cell_idx = self.id_to_cell[id.index as usize].load(Ordering::Acquire);

        if old_cell_idx == INVALID_CELL || new_cell_idx == old_cell_idx {
            return;
        }

        let old_cell = &self.cells[old_cell_idx];
        old_cell.remove(id);

        let new_cell = &self.cells[new_cell_idx];
        new_cell.add(id);

        self.id_to_cell[id.index as usize].store(new_cell_idx, Ordering::Release);
    }

    pub fn relocate(&self, ids: &[EntityId], positions_x: &[f32], positions_y: &[f32]) {
        ids.par_iter()
            .zip(positions_x.par_iter().zip(positions_y.par_iter()))
            .for_each(|(&id, (&pos_x, &pos_y))| {
                self._relocate(id, (pos_x, pos_y));
            });
    }

    pub fn insert(&self, id: EntityId, position: (f32, f32)) {
        let new_cell_idx = self.get_cell_index(position);

        let new_cell = &self.cells[new_cell_idx];
        new_cell.add(id);

        self.id_to_cell[id.index as usize].store(new_cell_idx, Ordering::Release);
    }

    pub fn remove(&self, id: EntityId) {
        let old_cell_idx = self.id_to_cell[id.index as usize].load(Ordering::Acquire);

        if old_cell_idx == INVALID_CELL {
            return;
        }

        let old_cell = &self.cells[old_cell_idx];
        old_cell.remove(id);

        self.id_to_cell[id.index as usize].store(INVALID_CELL, Ordering::Release);
    }
}
