use std::sync::OnceLock;

use parking_lot::RwLock;
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
    simd_lanes: usize,
    storage: EntityStorage,
    id_to_slot: Vec<usize>, // EntityId Index to Entity Slot Location in storage arrays
    free_ids: Vec<EntityId>,
}

static INSTANCE: OnceLock<RwLock<EntityManager>> = OnceLock::new();

impl EntityManager {
    pub fn get_instance() -> &'static RwLock<EntityManager> {
        INSTANCE.get_or_init(|| RwLock::new(EntityManager::new()))
    }

    fn new() -> Self {
        let config = Config::get_instance();

        Self {
            simd_lanes: config.simd_lanes as usize,
            storage: EntityStorage::new(config.max_entities),
            id_to_slot: (0..config.max_entities).map(|_| INVALID_SLOT).collect(),
            free_ids: Vec::new(),
        }
    }

    #[inline]
    pub fn slot(&self, id: EntityId) -> Option<usize> {
        let idx = self.id_to_slot[id.index as usize];
        if idx == INVALID_SLOT {
            return None;
        }

        // Stale EntityId / generation mismatch.
        if self.storage.id[idx] != id {
            return None;
        }

        Some(idx)
    }

    #[inline]
    pub fn get_ids(&self) -> &[EntityId] {
        &self.storage.id
    }
    #[inline]
    pub fn get_positions_x(&self) -> &[f32] {
        &self.storage.position_x
    }
    #[inline]
    pub fn get_positions_y(&self) -> &[f32] {
        &self.storage.position_y
    }
    #[inline]
    pub fn get_velocities_x(&self) -> &[f32] {
        &self.storage.velocity_x
    }
    #[inline]
    pub fn get_velocities_y(&self) -> &[f32] {
        &self.storage.velocity_y
    }
    // #[inline]
    // pub fn get_facings_x(&self) -> &[f32] {
    //     &self.storage.facing_x
    // }
    // #[inline]
    // pub fn get_facings_y(&self) -> &[f32] {
    //     &self.storage.facing_y
    // }
    // #[inline]
    // pub fn get_forces_x(&self) -> &[f32] {
    //     &self.storage.force_x
    // }
    // #[inline]
    // pub fn get_forces_y(&self) -> &[f32] {
    //     &self.storage.force_y
    // }
    // #[inline]
    // pub fn get_masses(&self) -> &[f32] {
    //     &self.storage.mass
    // }
    // #[inline]
    // pub fn get_inv_masses(&self) -> &[f32] {
    //     &self.storage.inv_mass
    // }
    #[inline]
    pub fn get_sizes(&self) -> &[f32] {
        &self.storage.size
    }
    // #[inline]
    // pub fn get_sensing_radii(&self) -> &[f32] {
    //     &self.storage.sensing_radius
    // }

    // #[inline]
    // pub fn get_entity_position(&self, id: EntityId) -> Option<(f32, f32)> {
    //     let idx = self.slot(id)?;
    //     Some((self.storage.position_x[idx], self.storage.position_y[idx]))
    // }

    #[inline]
    pub fn get_entity_velocity(&self, id: EntityId) -> Option<(f32, f32)> {
        let idx = self.slot(id)?;
        Some((self.storage.velocity_x[idx], self.storage.velocity_y[idx]))
    }

    // #[inline]
    // pub fn get_entity_size(&self, id: EntityId) -> Option<f32> {
    //     let idx = self.slot(id)?;
    //     Some(self.storage.size[idx])
    // }

    #[inline]
    pub fn get_entity_inv_mass(&self, id: EntityId) -> Option<f32> {
        let idx = self.slot(id)?;
        Some(self.storage.inv_mass[idx])
    }

    pub fn mut_entity_position(&mut self, id: EntityId, delta: (f32, f32)) {
        if let Some(idx) = self.slot(id) {
            self.storage.position_x[idx] += delta.0;
            self.storage.position_y[idx] += delta.1;
        }
    }

    pub fn mut_entity_velocity(&mut self, id: EntityId, delta: (f32, f32)) {
        if let Some(idx) = self.slot(id) {
            self.storage.velocity_x[idx] += delta.0;
            self.storage.velocity_y[idx] += delta.1;
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
    ) -> EntityId {
        assert!(self.storage.len() < Config::get_instance().max_entities);

        let id = match self.free_ids.pop() {
            Some(mut id) => id.update_gen(),
            None => {
                let last = self.storage.len() as u32;
                EntityId::new(last)
            }
        };

        let slot = self.storage.insert(
            id.clone(),
            position,
            velocity,
            facing,
            force,
            mass,
            size,
            sensing_radius,
        );

        self.id_to_slot[id.index as usize] = slot;

        id
    }

    // pub fn remove(&mut self, id: EntityId) -> bool {
    //     let idx = self.id_to_slot[id.index as usize];
    //     if idx == INVALID_SLOT {
    //         return false;
    //     }

    //     // Stale EntityId / generation mismatch.
    //     if self.storage.id[idx] != id {
    //         return false;
    //     }

    //     let (removed_id, swapped_id) = self.storage.remove(idx);

    //     if let Some(removed_id) = removed_id {
    //         self.id_to_slot[removed_id.index as usize] = INVALID_SLOT;
    //         self.free_ids.push(removed_id);
    //     }

    //     if let Some(swapped_id) = swapped_id {
    //         self.id_to_slot[swapped_id.index as usize] = idx;
    //     }

    //     true
    // }

    pub fn update_velocities(&mut self, dt: f32) {
        let dt_simd = f32x16::splat(dt);

        let remainder = self.storage.position_x.len() % self.simd_lanes;
        let simd_len = self.storage.position_x.len() - remainder;

        let (vx_simd, vx_tail) = self.storage.velocity_x.split_at_mut(simd_len);
        let (vy_simd, vy_tail) = self.storage.velocity_y.split_at_mut(simd_len);

        let (fx_simd, fx_tail) = self.storage.force_x.split_at_mut(simd_len);
        let (fy_simd, fy_tail) = self.storage.force_y.split_at_mut(simd_len);

        let (im_simd, im_tail) = self.storage.inv_mass.split_at(simd_len);

        vx_simd
            .par_chunks_exact_mut(self.simd_lanes)
            .zip(vy_simd.par_chunks_exact_mut(self.simd_lanes))
            .zip(fx_simd.par_chunks_exact_mut(self.simd_lanes))
            .zip(fy_simd.par_chunks_exact_mut(self.simd_lanes))
            .zip(im_simd.par_chunks_exact(self.simd_lanes))
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

        let remainder = self.storage.position_x.len() % self.simd_lanes;
        let simd_len = self.storage.position_x.len() - remainder;

        let (px_simd, px_tail) = self.storage.position_x.split_at_mut(simd_len);
        let (py_simd, py_tail) = self.storage.position_y.split_at_mut(simd_len);

        let (vx_simd, vx_tail) = self.storage.velocity_x.split_at(simd_len);
        let (vy_simd, vy_tail) = self.storage.velocity_y.split_at(simd_len);

        px_simd
            .par_chunks_exact_mut(self.simd_lanes)
            .zip(py_simd.par_chunks_exact_mut(self.simd_lanes))
            .zip(vx_simd.par_chunks_exact(self.simd_lanes))
            .zip(vy_simd.par_chunks_exact(self.simd_lanes))
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
