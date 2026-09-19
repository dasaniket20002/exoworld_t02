use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};
use wide::f32x16;

use crate::{
    entities::{entity_id::EntityId, entity_storage::EntityStorage},
    global::config::Config,
};

pub const INVALID_SLOT: usize = usize::MAX;

pub struct EntityManager {
    storage: EntityStorage,
    id_to_slot: Vec<usize>, // EntityId Index to Entity Slot Location in storage arrays
    free_ids: Vec<EntityId>,
}

impl EntityManager {
    pub fn new() -> Self {
        let config = Config::get_instance();

        Self {
            storage: EntityStorage::new(config.max_entities),
            id_to_slot: (0..config.max_entities).map(|_| INVALID_SLOT).collect(),
            free_ids: Vec::new(),
        }
    }

    pub fn insert(
        &mut self,
        position: (f32, f32),
        velocity: (f32, f32),
        facing: Option<(f32, f32)>,
        force: Option<(f32, f32)>,
        mass: f32,
        size: f32,
        sensing_radius: f32,
    ) {
        if self.storage.len() >= Config::get_instance().max_entities {
            return;
        }

        let id = match self.free_ids.pop() {
            Some(mut id) => id.update_gen(),
            None => {
                let last = self.storage.len() as u32;
                EntityId::new(last)
            }
        };

        let entity_id_index = id.index as usize;

        let slot = self.storage.insert(
            id,
            position,
            velocity,
            facing,
            force,
            mass,
            size,
            sensing_radius,
        );

        self.id_to_slot[entity_id_index] = slot;
    }

    pub fn remove(&mut self, id: EntityId) -> bool {
        let idx = self.id_to_slot[id.index as usize];
        if idx == INVALID_SLOT {
            return false;
        }

        // Stale EntityId / generation mismatch.
        if self.storage.id[idx] != id {
            return false;
        }

        let (removed_id, swapped_id) = self.storage.remove(idx);

        if let Some(removed_id) = removed_id {
            self.id_to_slot[removed_id.index as usize] = INVALID_SLOT;
            self.free_ids.push(removed_id);
        }

        if let Some(swapped_id) = swapped_id {
            self.id_to_slot[swapped_id.index as usize] = idx;
        }

        true
    }

    #[inline]
    pub fn contains(&self, id: EntityId) -> bool {
        let idx = self.id_to_slot[id.index as usize];
        if idx == INVALID_SLOT {
            return false;
        }

        // Stale EntityId / generation mismatch.
        if self.storage.id[idx] != id {
            return false;
        }

        true
    }

    pub fn update_velocities(&mut self, dt: f32) {
        let dt_simd = f32x16::splat(dt);
        let simd_lanes = Config::get_instance().simd_lanes as usize;

        let remainder = self.storage.position_x.len() % simd_lanes;
        let simd_len = self.storage.position_x.len() - remainder;

        let (vx_simd, vx_tail) = self.storage.velocity_x.split_at_mut(simd_len);
        let (vy_simd, vy_tail) = self.storage.velocity_y.split_at_mut(simd_len);

        let (fx_simd, fx_tail) = self.storage.force_x.split_at_mut(simd_len);
        let (fy_simd, fy_tail) = self.storage.force_y.split_at_mut(simd_len);

        let (im_simd, im_tail) = self.storage.inv_mass.split_at(simd_len);

        vx_simd
            .par_chunks_exact_mut(simd_lanes)
            .zip(vy_simd.par_chunks_exact_mut(simd_lanes))
            .zip(fx_simd.par_chunks_exact_mut(simd_lanes))
            .zip(fy_simd.par_chunks_exact_mut(simd_lanes))
            .zip(im_simd.par_chunks_exact(simd_lanes))
            .for_each(|((((vx, vy), fx), fy), im)| {
                let vx_s = f32x16::from(&*vx);
                let vy_s = f32x16::from(&*vy);

                let fx_s = f32x16::from(&*fx);
                let fy_s = f32x16::from(&*fy);

                let im_s = f32x16::from(im);

                let acc_x = fx_s * im_s;
                let acc_y = fy_s * im_s;

                let n_vx = acc_x.mul_add(dt_simd, vx_s);
                let n_vy = acc_y.mul_add(dt_simd, vy_s);

                vx.copy_from_slice(n_vx.as_array());
                vy.copy_from_slice(n_vy.as_array());

                fx.copy_from_slice(f32x16::ZERO.as_array());
                fy.copy_from_slice(f32x16::ZERO.as_array());
            });

        for i in 0..remainder {
            let acc_x = fx_tail[i] * im_tail[i];
            let acc_y = fy_tail[i] * im_tail[i];

            vx_tail[i] += acc_x * dt;
            vy_tail[i] += acc_y * dt;

            fx_tail[i] = 0.0;
            fy_tail[i] = 0.0;
        }
    }

    pub fn update_positions(&mut self, dt: f32) {
        let dt_simd = f32x16::splat(dt);
        let simd_lanes = Config::get_instance().simd_lanes as usize;

        let remainder = self.storage.position_x.len() % simd_lanes;
        let simd_len = self.storage.position_x.len() - remainder;

        let (px_simd, px_tail) = self.storage.position_x.split_at_mut(simd_len);
        let (py_simd, py_tail) = self.storage.position_y.split_at_mut(simd_len);

        let (vx_simd, vx_tail) = self.storage.velocity_x.split_at(simd_len);
        let (vy_simd, vy_tail) = self.storage.velocity_y.split_at(simd_len);

        px_simd
            .par_chunks_exact_mut(simd_lanes)
            .zip(py_simd.par_chunks_exact_mut(simd_lanes))
            .zip(vx_simd.par_chunks_exact(simd_lanes))
            .zip(vy_simd.par_chunks_exact(simd_lanes))
            .for_each(|(((px, py), vx), vy)| {
                let px_v = f32x16::from(&*px);
                let py_v = f32x16::from(&*py);

                let vx_v = f32x16::from(vx);
                let vy_v = f32x16::from(vy);

                let new_px = vx_v.mul_add(dt_simd, px_v);
                let new_py = vy_v.mul_add(dt_simd, py_v);

                px.copy_from_slice(new_px.as_array());
                py.copy_from_slice(new_py.as_array());
            });

        // Scalar tail.
        for i in 0..remainder {
            px_tail[i] += vx_tail[i] * dt;
            py_tail[i] += vy_tail[i] * dt;
        }
    }
}
