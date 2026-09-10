use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::{ParallelSlice, ParallelSliceMut},
};
use wide::f32x16;

use crate::{entities::entity_id::EntityId, global::config::Config};

const SIMD_LANES: usize = 16;

#[derive(Clone, Copy, Debug)]
struct EntitySlot {
    generation: u32,
    dense_index: u32,
    alive: bool,
}

pub struct EntitiesSoa {
    // Stable ID corresponding to each dense row.
    pub entity_ids: Vec<EntityId>,

    // Component columns.
    pub position_x: Vec<f32>,
    pub position_y: Vec<f32>,

    pub velocity_x: Vec<f32>,
    pub velocity_y: Vec<f32>,

    pub force_x: Vec<f32>,
    pub force_y: Vec<f32>,

    pub mass: Vec<f32>,
    pub inv_mass: Vec<f32>,

    pub radius: Vec<f32>,

    // Stable-ID table.
    slots: Vec<EntitySlot>,

    // Reusable stable-ID slots.
    free_slots: Vec<u32>,

    max_entities: usize,
}

impl EntitiesSoa {
    pub fn new() -> Self {
        let config = Config::get_instance();
        Self {
            entity_ids: Vec::with_capacity(config.max_entities),

            position_x: Vec::with_capacity(config.max_entities),
            position_y: Vec::with_capacity(config.max_entities),

            velocity_x: Vec::with_capacity(config.max_entities),
            velocity_y: Vec::with_capacity(config.max_entities),

            force_x: Vec::with_capacity(config.max_entities),
            force_y: Vec::with_capacity(config.max_entities),

            mass: Vec::with_capacity(config.max_entities),
            inv_mass: Vec::with_capacity(config.max_entities),

            radius: Vec::with_capacity(config.max_entities),

            slots: Vec::with_capacity(config.max_entities),
            free_slots: Vec::with_capacity(config.max_entities),

            max_entities: config.max_entities,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entity_ids.len()
    }

    #[inline]
    pub fn dense_index(&self, entity: EntityId) -> Option<usize> {
        let slot = self.slots.get(entity.index as usize)?;

        if !slot.alive || slot.generation != entity.generation {
            return None;
        }

        Some(slot.dense_index as usize)
    }

    #[inline]
    pub fn contains(&self, entity: EntityId) -> bool {
        self.dense_index(entity).is_some()
    }

    pub fn add(
        &mut self,
        position_x: f32,
        position_y: f32,
        velocity_x: f32,
        velocity_y: f32,
        force_x: f32,
        force_y: f32,
        mass: f32,
        radius: f32,
    ) -> (EntityId, u32) {
        assert!(
            self.len() < self.max_entities,
            "maximum entity count reached"
        );

        let dense_index = self.len() as u32;

        let entity = if let Some(index) = self.free_slots.pop() {
            let slot = &mut self.slots[index as usize];

            debug_assert!(!slot.alive);

            slot.alive = true;
            slot.dense_index = dense_index;

            EntityId::new(index, slot.generation)
        } else {
            let index = self.slots.len();

            assert!(index <= u32::MAX as usize, "too many stable entity IDs");

            self.slots.push(EntitySlot {
                generation: 0,
                dense_index,
                alive: true,
            });

            EntityId::new(index as u32, 0)
        };

        self.entity_ids.push(entity);

        self.position_x.push(position_x);
        self.position_y.push(position_y);

        self.velocity_x.push(velocity_x);
        self.velocity_y.push(velocity_y);

        self.force_x.push(force_x);
        self.force_y.push(force_y);

        self.mass.push(mass);
        self.inv_mass.push(1.0 / mass);

        self.radius.push(radius);

        (entity, dense_index)
    }

    pub fn remove(&mut self, entity: EntityId) -> Option<(EntityId, u32)> {
        let stable_index = entity.index as usize;

        let slot = self.slots.get(stable_index).copied()?;

        if !slot.alive || slot.generation != entity.generation {
            return None;
        }

        let dense_index = slot.dense_index as usize;
        let last_dense_index = self.entity_ids.len() - 1;

        let moved_entity = if dense_index != last_dense_index {
            let moved = self.entity_ids[last_dense_index];

            self.entity_ids.swap(dense_index, last_dense_index);

            self.position_x.swap(dense_index, last_dense_index);

            self.position_y.swap(dense_index, last_dense_index);

            self.velocity_x.swap(dense_index, last_dense_index);

            self.velocity_y.swap(dense_index, last_dense_index);

            self.force_x.swap(dense_index, last_dense_index);

            self.force_y.swap(dense_index, last_dense_index);

            self.mass.swap(dense_index, last_dense_index);

            self.inv_mass.swap(dense_index, last_dense_index);

            self.radius.swap(dense_index, last_dense_index);

            self.slots[moved.index as usize].dense_index = dense_index as u32;

            Some((moved, dense_index as u32))
        } else {
            None
        };

        self.entity_ids.pop();

        self.position_x.pop();
        self.position_y.pop();

        self.velocity_x.pop();
        self.velocity_y.pop();

        self.force_x.pop();
        self.force_y.pop();

        self.mass.pop();
        self.inv_mass.pop();

        self.radius.pop();

        let slot = &mut self.slots[stable_index];

        slot.alive = false;
        slot.generation = slot.generation.wrapping_add(1);

        self.free_slots.push(entity.index);

        moved_entity
    }

    pub fn update_velocities(&mut self, dt: f32) {
        let dt_simd = f32x16::splat(dt);

        let remainder = self.position_x.len() % SIMD_LANES;
        let simd_len = self.position_x.len() - remainder;

        let (vx_simd, vx_tail) = self.velocity_x.split_at_mut(simd_len);
        let (vy_simd, vy_tail) = self.velocity_y.split_at_mut(simd_len);

        let (fx_simd, fx_tail) = self.force_x.split_at_mut(simd_len);
        let (fy_simd, fy_tail) = self.force_y.split_at_mut(simd_len);

        let (im_simd, im_tail) = self.inv_mass.split_at(simd_len);

        vx_simd
            .par_chunks_exact_mut(SIMD_LANES)
            .zip(vy_simd.par_chunks_exact_mut(SIMD_LANES))
            .zip(fx_simd.par_chunks_exact_mut(SIMD_LANES))
            .zip(fy_simd.par_chunks_exact_mut(SIMD_LANES))
            .zip(im_simd.par_chunks_exact(SIMD_LANES))
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

        let remainder = self.position_x.len() % SIMD_LANES;
        let simd_len = self.position_x.len() - remainder;

        let (px_simd, px_tail) = self.position_x.split_at_mut(simd_len);
        let (py_simd, py_tail) = self.position_y.split_at_mut(simd_len);

        let (vx_simd, vx_tail) = self.velocity_x.split_at(simd_len);
        let (vy_simd, vy_tail) = self.velocity_y.split_at(simd_len);

        px_simd
            .par_chunks_exact_mut(SIMD_LANES)
            .zip(py_simd.par_chunks_exact_mut(SIMD_LANES))
            .zip(vx_simd.par_chunks_exact(SIMD_LANES))
            .zip(vy_simd.par_chunks_exact(SIMD_LANES))
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

    pub fn apply_collision_deltas(
        &mut self,
        delta_velocity_x: &[f32],
        delta_velocity_y: &[f32],
        delta_position_x: &[f32],
        delta_position_y: &[f32],
    ) {
        debug_assert_eq!(self.len(), delta_velocity_x.len());

        debug_assert_eq!(self.len(), delta_velocity_y.len());

        debug_assert_eq!(self.len(), delta_position_x.len());

        debug_assert_eq!(self.len(), delta_position_y.len());

        let Self {
            position_x,
            position_y,

            velocity_x,
            velocity_y,
            ..
        } = self;

        position_x
            .par_iter_mut()
            .zip(position_y.par_iter_mut())
            .zip(velocity_x.par_iter_mut())
            .zip(velocity_y.par_iter_mut())
            .zip(delta_position_x.par_iter())
            .zip(delta_position_y.par_iter())
            .zip(delta_velocity_x.par_iter())
            .zip(delta_velocity_y.par_iter())
            .for_each(|(((((((px, py), vx), vy), dpx), dpy), dvx), dvy)| {
                *px += *dpx;
                *py += *dpy;

                *vx += *dvx;
                *vy += *dvy;
            });
    }
}
