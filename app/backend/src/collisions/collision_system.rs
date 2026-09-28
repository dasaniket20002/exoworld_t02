use std::sync::OnceLock;

use parking_lot::{RwLock, RwLockWriteGuard};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    entities::{entity_id::EntityId, entity_manager::EntityManager},
    global::config::Config,
    spatial_db::grid_manager::GridManager,
};

static INSTANCE: OnceLock<RwLock<CollisionSystem>> = OnceLock::new();

pub struct CollisionContact {
    pub id_a: EntityId,
    pub id_b: EntityId,

    pub normal_x: f32,
    pub normal_y: f32,

    pub penetration: f32,
}

pub struct CollisionSystem {
    contacts: Vec<CollisionContact>,
}

impl CollisionSystem {
    pub fn get_instance() -> &'static RwLock<CollisionSystem> {
        INSTANCE.get_or_init(|| RwLock::new(CollisionSystem::new()))
    }

    pub fn new() -> Self {
        Self {
            contacts: Vec::new(),
        }
    }

    #[inline(always)]
    fn detect_pair(
        id_a: EntityId,
        id_b: EntityId,
        positions_x: &[f32],
        positions_y: &[f32],
        velocities_x: &[f32],
        velocities_y: &[f32],
        sizes: &[f32],
    ) -> Option<CollisionContact> {
        let idx_a = id_a.index as usize;
        let idx_b = id_b.index as usize;

        let radius = sizes[idx_a] + sizes[idx_b];

        if radius <= 0.0 {
            return None;
        }

        let dx = positions_x[idx_b] - positions_x[idx_a];
        let dy = positions_y[idx_b] - positions_y[idx_a];

        let distance_squared = dx * dx + dy * dy;
        let radius_squared = radius * radius;

        if distance_squared >= radius_squared {
            return None;
        }

        let (normal_x, normal_y, distance) = if distance_squared > f32::EPSILON {
            let distance = distance_squared.sqrt();
            let inverse_distance = 1.0 / distance;

            (dx * inverse_distance, dy * inverse_distance, distance)
        } else {
            let rvx = velocities_x[idx_b] - velocities_x[idx_a];
            let rvy = velocities_y[idx_b] - velocities_y[idx_a];

            let rv_squared = rvx * rvx + rvy * rvy;

            if rv_squared > f32::EPSILON {
                let rv_length = rv_squared.sqrt();
                let inverse_length = 1.0 / rv_length;

                (rvx * inverse_length, rvy * inverse_length, 0.0)
            } else {
                let sign = if id_a.index <= id_b.index { 1.0 } else { -1.0 };

                (sign, 0.0, 0.0)
            }
        };

        Some(CollisionContact {
            id_a,
            id_b,
            normal_x,
            normal_y,
            penetration: radius - distance,
        })
    }

    pub fn detect_collisions(&mut self) {
        let grid = GridManager::get_instance().read();
        let entities = EntityManager::get_instance().read();

        let cells = grid.get_cells();
        let active_cells = grid.get_active_cells();
        let cell_count_axis = grid.get_cell_count_axis();

        let positions_x = entities.get_positions_x();
        let positions_y = entities.get_positions_y();
        let velocities_x = entities.get_velocities_x();
        let velocities_y = entities.get_velocities_y();
        let sizes = entities.get_sizes();

        self.contacts = active_cells
            .par_iter()
            .fold(
                || Vec::with_capacity(32),
                |mut contacts, &cell_index| {
                    let cell_x = cell_index % cell_count_axis;
                    let cell_y = cell_index / cell_count_axis;

                    let current = cells[cell_index].get_entities();

                    // Same cell.
                    for i in 0..current.len() {
                        let a = current[i];

                        for &b in &current[i + 1..] {
                            if let Some(contact) = Self::detect_pair(
                                a,
                                b,
                                positions_x,
                                positions_y,
                                velocities_x,
                                velocities_y,
                                sizes,
                            ) {
                                contacts.push(contact);
                            }
                        }
                    }

                    // Right.
                    if cell_x + 1 < cell_count_axis {
                        let neighbor_index = cell_index + 1;

                        if grid.is_active_cell(neighbor_index) {
                            let neighbor = cells[neighbor_index].get_entities();

                            for &a in current.iter() {
                                for &b in neighbor.iter() {
                                    if let Some(contact) = Self::detect_pair(
                                        a,
                                        b,
                                        positions_x,
                                        positions_y,
                                        velocities_x,
                                        velocities_y,
                                        sizes,
                                    ) {
                                        contacts.push(contact);
                                    }
                                }
                            }
                        }
                    }

                    // Down-left.
                    if cell_y + 1 < cell_count_axis && cell_x > 0 {
                        let neighbor_index = cell_index + cell_count_axis - 1;

                        if grid.is_active_cell(neighbor_index) {
                            let neighbor = cells[neighbor_index].get_entities();

                            for &a in current.iter() {
                                for &b in neighbor.iter() {
                                    if let Some(contact) = Self::detect_pair(
                                        a,
                                        b,
                                        positions_x,
                                        positions_y,
                                        velocities_x,
                                        velocities_y,
                                        sizes,
                                    ) {
                                        contacts.push(contact);
                                    }
                                }
                            }
                        }
                    }

                    // Down.
                    if cell_y + 1 < cell_count_axis {
                        let neighbor_index = cell_index + cell_count_axis;

                        if grid.is_active_cell(neighbor_index) {
                            let neighbor = cells[neighbor_index].get_entities();

                            for &a in current.iter() {
                                for &b in neighbor.iter() {
                                    if let Some(contact) = Self::detect_pair(
                                        a,
                                        b,
                                        positions_x,
                                        positions_y,
                                        velocities_x,
                                        velocities_y,
                                        sizes,
                                    ) {
                                        contacts.push(contact);
                                    }
                                }
                            }
                        }
                    }

                    // Down-right.
                    if cell_y + 1 < cell_count_axis && cell_x + 1 < cell_count_axis {
                        let neighbor_index = cell_index + cell_count_axis + 1;

                        if grid.is_active_cell(neighbor_index) {
                            let neighbor = cells[neighbor_index].get_entities();

                            for &a in current.iter() {
                                for &b in neighbor.iter() {
                                    if let Some(contact) = Self::detect_pair(
                                        a,
                                        b,
                                        positions_x,
                                        positions_y,
                                        velocities_x,
                                        velocities_y,
                                        sizes,
                                    ) {
                                        contacts.push(contact);
                                    }
                                }
                            }
                        }
                    }

                    contacts
                },
            )
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            });
    }

    fn apply_position_correction(
        &self,
        entity_manager_write: &mut RwLockWriteGuard<'_, EntityManager>,
        contact: &CollisionContact,
        inv_mass_sum: f32,
        inv_mass_a: f32,
        inv_mass_b: f32,
    ) {
        let config = Config::get_instance();
        let correction_depth = (contact.penetration - config.penetration_slop).max(0.0);
        let correction_magnitude = correction_depth * config.correction_percent / inv_mass_sum;

        let correction_x = contact.normal_x * correction_magnitude;
        let correction_y = contact.normal_y * correction_magnitude;

        let pos_delta_a = (-correction_x * inv_mass_a, -correction_y * inv_mass_a);
        let pos_delta_b = (correction_x * inv_mass_b, correction_y * inv_mass_b);

        entity_manager_write.mut_entity_position(contact.id_a, pos_delta_a);
        entity_manager_write.mut_entity_position(contact.id_b, pos_delta_b);
    }

    fn apply_impulse(
        &self,
        entity_manager_write: &mut RwLockWriteGuard<'_, EntityManager>,
        contact: &CollisionContact,
        inv_mass_sum: f32,
        inv_mass_a: f32,
        inv_mass_b: f32,
    ) {
        let (va, vb) = {
            let entity_manager_read = EntityManager::get_instance().read();
            let va = entity_manager_read
                .get_entity_velocity(contact.id_a)
                .unwrap();
            let vb = entity_manager_read
                .get_entity_velocity(contact.id_b)
                .unwrap();

            (va, vb)
        };

        let relative_velocity_x = vb.0 - va.0;
        let relative_velocity_y = vb.1 - va.1;

        let velocity_along_normal =
            relative_velocity_x * contact.normal_x + relative_velocity_y * contact.normal_y;

        // Already separating.
        if velocity_along_normal > 0.0 {
            return;
        }

        let config = Config::get_instance();

        let impulse_magnitude =
            -(1.0 + config.collision_restitution) * velocity_along_normal / inv_mass_sum;

        let impulse_x = impulse_magnitude * contact.normal_x;
        let impulse_y = impulse_magnitude * contact.normal_y;

        let va_delta = (-impulse_x * inv_mass_a, -impulse_y * inv_mass_a);
        let vb_delta = (impulse_x * inv_mass_b, impulse_y * inv_mass_b);

        entity_manager_write.mut_entity_velocity(contact.id_a, va_delta);
        entity_manager_write.mut_entity_velocity(contact.id_b, vb_delta);
    }

    pub fn resolve_collisions(&self) {
        let mut entity_manager = EntityManager::get_instance().write();

        for contact in &self.contacts {
            let (inv_mass_a, inv_mass_b) = {
                let inv_mass_a = entity_manager
                    .get_entity_inv_mass(contact.id_a)
                    .unwrap_or(0.0);
                let inv_mass_b = entity_manager
                    .get_entity_inv_mass(contact.id_b)
                    .unwrap_or(0.0);
                (inv_mass_a, inv_mass_b)
            };

            let inv_mass_sum = inv_mass_a + inv_mass_b;

            if inv_mass_sum <= f32::EPSILON {
                return;
            }

            self.apply_position_correction(
                &mut entity_manager,
                contact,
                inv_mass_sum,
                inv_mass_a,
                inv_mass_b,
            );
            self.apply_impulse(
                &mut entity_manager,
                contact,
                inv_mass_sum,
                inv_mass_a,
                inv_mass_b,
            );
        }
    }
}
