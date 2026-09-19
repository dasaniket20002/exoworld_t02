use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use wide::f32x16;

use crate::{
    entities::entities_soa::EntitiesSoa, global::config::Config,
    spatial_db::grid_manager::UniformGrid,
};

const LANES: usize = 16;

#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub a: u32,
    pub b: u32,
}

/*
    Per-contact data calculated during each solver iteration.

    ix, iy:
        impulse vector WITHOUT inverse-mass multiplication.

    px, py:
        positional correction vector WITHOUT inverse-mass
        multiplication.

    Keeping these in a reusable array means the geometry/normal/
    penetration calculation happens ONCE per contact per iteration,
    not once for A reduction and once for B reduction.
*/
#[derive(Clone, Copy, Default)]
struct ContactDelta {
    ix: f32,
    iy: f32,

    px: f32,
    py: f32,
}

pub struct CollisionSystem {
    /*
        Collision pairs.

        Dense SoA indices, NOT EntityId.
    */
    contacts: Vec<Contact>,

    /*
        Cached per-contact solver data.
    */
    contact_deltas: Vec<ContactDelta>,

    /*
        Contact indexes grouped by A.

        a_offsets[e] ..
        a_offsets[e + 1]

        contains all contacts where entity e is A.
    */
    by_a: Vec<u32>,
    a_offsets: Vec<usize>,
    a_cursors: Vec<usize>,

    /*
        Same for B.
    */
    by_b: Vec<u32>,
    b_offsets: Vec<usize>,
    b_cursors: Vec<usize>,

    /*
        Cached per-entity solver accumulation.

        These are intentionally retained across frames.
    */
    delta_velocity_x: Vec<f32>,
    delta_velocity_y: Vec<f32>,

    delta_position_x: Vec<f32>,
    delta_position_y: Vec<f32>,

    pub solver_iterations: usize,

    pub restitution: f32,

    pub penetration_slop: f32,

    pub position_correction: f32,
}

impl CollisionSystem {
    pub fn new(solver_iterations: usize) -> Self {
        let config = Config::get_instance();
        Self {
            contacts: Vec::with_capacity(config.max_entities),

            contact_deltas: Vec::with_capacity(config.max_entities),

            by_a: Vec::with_capacity(config.max_entities),

            a_offsets: vec![0; config.max_entities + 1],

            a_cursors: vec![0; config.max_entities],

            by_b: Vec::with_capacity(config.max_entities),

            b_offsets: vec![0; config.max_entities + 1],

            b_cursors: vec![0; config.max_entities],

            delta_velocity_x: vec![0.0; config.max_entities],

            delta_velocity_y: vec![0.0; config.max_entities],

            delta_position_x: vec![0.0; config.max_entities],

            delta_position_y: vec![0.0; config.max_entities],

            solver_iterations: solver_iterations.max(1),

            restitution: config.collision_restitution.clamp(0.0, 1.0),

            penetration_slop: 0.001,

            position_correction: 0.8,
        }
    }

    #[inline]
    pub fn contacts(&self) -> &[Contact] {
        &self.contacts
    }

    pub fn detect(&mut self, grid: &UniformGrid, entities: &EntitiesSoa) {
        self.contacts.clear();

        if entities.len() == 0 {
            self.clear_groups();
            return;
        }

        let cells = grid.cells();

        let num_cells_axis = grid.num_cells_axis;

        let px = &entities.position_x;
        let py = &entities.position_y;
        let radius = &entities.radius;

        /*
            Exactly the same broad-phase traversal as the SIMD
            implementation.

            There is deliberately no sorting here.
            The half-neighborhood guarantees each pair is tested once.
        */
        let detected = cells
            .par_iter()
            .enumerate()
            .fold(
                || Vec::<Contact>::with_capacity(16),
                |mut local, (cell_index, cell)| {
                    let current = &cell.entities;

                    /*
                        ------------------------------------------------
                        SAME CELL
                        ------------------------------------------------
                    */

                    for i in 0..current.len() {
                        let a = current[i] as usize;

                        let b_candidates = &current[i + 1..];

                        Self::test_candidates_scalar(&mut local, a, b_candidates, px, py, radius);
                    }

                    /*
                        ------------------------------------------------
                        HALF NEIGHBORHOOD
                        ------------------------------------------------

                        right
                        up-left
                        up
                        up-right
                    */
                    let cell_index = cell_index as u32;

                    let cx = cell_index % num_cells_axis;

                    let cy = cell_index / num_cells_axis;

                    let neighbors = [(1i32, 0i32), (-1, 1), (0, 1), (1, 1)];

                    for &(dx, dy) in &neighbors {
                        let nx = cx as i32 + dx;

                        let ny = cy as i32 + dy;

                        if nx < 0
                            || ny < 0
                            || nx >= num_cells_axis as i32
                            || ny >= num_cells_axis as i32
                        {
                            continue;
                        }

                        let neighbor = ny as u32 * num_cells_axis + nx as u32;

                        let other = &cells[neighbor as usize].entities;

                        if other.is_empty() {
                            continue;
                        }

                        for &a_dense in current {
                            Self::test_candidates_scalar(
                                &mut local,
                                a_dense as usize,
                                other,
                                px,
                                py,
                                radius,
                            );
                        }
                    }

                    local
                },
            )
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            });

        self.contacts = detected;

        /*
            Keep the exact same contact adjacency construction
            for both benchmark paths.
        */
        self.build_groups(entities.len());
    }

    #[inline(always)]
    fn test_candidates_scalar(
        output: &mut Vec<Contact>,
        a: usize,
        candidates: &[u32],
        px: &[f32],
        py: &[f32],
        radius: &[f32],
    ) {
        let ax = px[a];
        let ay = py[a];
        let ar = radius[a];

        for &b_dense in candidates {
            let b = b_dense as usize;

            let dx = px[b] - ax;
            let dy = py[b] - ay;

            let sum_r = radius[b] + ar;

            let dist_sq = dx * dx + dy * dy;

            /*
                Strict overlap.

                This matches the SIMD version exactly:
                    dist_sq < (ra + rb)^2
            */
            if dist_sq < sum_r * sum_r {
                output.push(Contact {
                    a: a as u32,
                    b: b_dense,
                });
            }
        }
    }

    /*
        SIMD circle-circle narrow phase.

        The B side is gathered into 16 temporary arrays because
        grid membership is not contiguous in the SoA.

        The expensive distance comparison itself is SIMD.
    */
    #[inline]
    fn test_candidates_simd(
        output: &mut Vec<Contact>,
        a: usize,
        candidates: &[u32],

        px: &[f32],
        py: &[f32],
        radius: &[f32],
    ) {
        let ax = px[a];

        let ay = py[a];

        let ar = radius[a];

        let ax_v = f32x16::splat(ax);

        let ay_v = f32x16::splat(ay);

        let ar_v = f32x16::splat(ar);

        let mut start = 0;

        while start + LANES <= candidates.len() {
            let chunk = &candidates[start..start + LANES];

            let mut bx = [0.0f32; LANES];

            let mut by = [0.0f32; LANES];

            let mut br = [0.0f32; LANES];

            for lane in 0..LANES {
                let dense = chunk[lane] as usize;

                bx[lane] = px[dense];

                by[lane] = py[dense];

                br[lane] = radius[dense];
            }

            let bx_v = f32x16::new(bx);

            let by_v = f32x16::new(by);

            let br_v = f32x16::new(br);

            let dx = bx_v - ax_v;

            let dy = by_v - ay_v;

            let dist_sq = dx.mul_add(dx, dy * dy);

            let sum_r = br_v + ar_v;

            let collision = dist_sq.simd_lt(sum_r * sum_r);

            let mut mask = collision.to_bitmask();

            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;

                let b = chunk[lane];

                output.push(Contact { a: a as u32, b });

                mask &= mask - 1;
            }

            start += LANES;
        }

        /*
            Scalar tail.
        */
        for &b_dense in &candidates[start..] {
            let b = b_dense as usize;

            let dx = px[b] - ax;

            let dy = py[b] - ay;

            let sum_r = radius[b] + ar;

            let dist_sq = dx * dx + dy * dy;

            if dist_sq < sum_r * sum_r {
                output.push(Contact {
                    a: a as u32,
                    b: b_dense,
                });
            }
        }
    }

    fn clear_groups(&mut self) {
        self.by_a.clear();
        self.by_b.clear();

        self.a_offsets.fill(0);
        self.b_offsets.fill(0);
    }

    /*
        Builds contact adjacency without a comparison sort.

        Because entity indexes are dense 0..N-1, counting/bucketing
        is a natural fit.
    */
    fn build_groups(&mut self, entity_count: usize) {
        self.a_offsets[..=entity_count].fill(0);

        self.b_offsets[..=entity_count].fill(0);

        /*
            Count A/B contacts.
        */

        for contact in &self.contacts {
            self.a_offsets[contact.a as usize + 1] += 1;

            self.b_offsets[contact.b as usize + 1] += 1;
        }

        /*
            Prefix sums.
        */

        for i in 1..=entity_count {
            self.a_offsets[i] += self.a_offsets[i - 1];

            self.b_offsets[i] += self.b_offsets[i - 1];
        }

        self.by_a.resize(self.contacts.len(), 0);

        self.by_b.resize(self.contacts.len(), 0);

        /*
            Copy starts into cursors.
        */

        self.a_cursors[..entity_count].copy_from_slice(&self.a_offsets[..entity_count]);

        self.b_cursors[..entity_count].copy_from_slice(&self.b_offsets[..entity_count]);

        /*
            Scatter contact indexes.
        */

        for (contact_index, contact) in self.contacts.iter().enumerate() {
            let a = contact.a as usize;

            let a_slot = self.a_cursors[a];

            self.by_a[a_slot] = contact_index as u32;

            self.a_cursors[a] += 1;

            let b = contact.b as usize;

            let b_slot = self.b_cursors[b];

            self.by_b[b_slot] = contact_index as u32;

            self.b_cursors[b] += 1;
        }
    }

    pub fn solve(&mut self, entities: &mut EntitiesSoa) {
        if self.contacts.is_empty() {
            return;
        }

        let entity_count = entities.len();

        let contact_count = self.contacts.len();

        self.delta_velocity_x.resize(entity_count, 0.0);

        self.delta_velocity_y.resize(entity_count, 0.0);

        self.delta_position_x.resize(entity_count, 0.0);

        self.delta_position_y.resize(entity_count, 0.0);

        self.contact_deltas
            .resize(contact_count, ContactDelta::default());

        for iteration in 0..self.solver_iterations {
            /*
                Clear cached entity accumulators.

                These arrays are reused between frames.
            */
            self.clear_entity_deltas(entity_count);

            /*
                Calculate geometry/normal/impulse once
                for every contact.
            */
            self.calculate_contact_deltas(entities, iteration);

            /*
                Reduce the contribution from each contact
                onto A.

                Every entity owns one unique output slot.
            */
            self.reduce_a(entities);

            /*
                Then add B contributions.
            */
            self.reduce_b(entities);

            /*
                Apply all corrections in parallel.
            */
            entities.apply_collision_deltas(
                &self.delta_velocity_x,
                &self.delta_velocity_y,
                &self.delta_position_x,
                &self.delta_position_y,
            );
        }
    }

    fn clear_entity_deltas(&mut self, entity_count: usize) {
        self.delta_velocity_x[..entity_count]
            .par_iter_mut()
            .for_each(|v| *v = 0.0);

        self.delta_velocity_y[..entity_count]
            .par_iter_mut()
            .for_each(|v| *v = 0.0);

        self.delta_position_x[..entity_count]
            .par_iter_mut()
            .for_each(|v| *v = 0.0);

        self.delta_position_y[..entity_count]
            .par_iter_mut()
            .for_each(|v| *v = 0.0);
    }

    fn calculate_contact_deltas(&mut self, entities: &EntitiesSoa, iteration: usize) {
        let px = &entities.position_x;

        let py = &entities.position_y;

        let vx = &entities.velocity_x;

        let vy = &entities.velocity_y;

        let inv_mass = &entities.inv_mass;

        let radius = &entities.radius;

        let restitution = if iteration == 0 {
            self.restitution
        } else {
            0.0
        };

        let slop = self.penetration_slop;

        let correction_percent = self.position_correction;

        let contacts = &self.contacts;

        self.contact_deltas
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, output)| {
                let contact = contacts[index];

                let a = contact.a as usize;

                let b = contact.b as usize;

                let dx = px[b] - px[a];

                let dy = py[b] - py[a];

                let distance_sq = dx * dx + dy * dy;

                let combined_radius = radius[a] + radius[b];

                if distance_sq >= combined_radius * combined_radius {
                    *output = ContactDelta::default();

                    return;
                }

                let distance = distance_sq.sqrt();

                let (nx, ny) = if distance > 1.0e-6 {
                    (dx / distance, dy / distance)
                } else {
                    /*
                        Deterministic normal
                        for exactly overlapping
                        centers.
                    */
                    if a < b { (1.0, 0.0) } else { (-1.0, 0.0) }
                };

                let penetration = combined_radius - distance;

                let rvx = vx[b] - vx[a];

                let rvy = vy[b] - vy[a];

                let normal_velocity = rvx * nx + rvy * ny;

                let inv_mass_sum = inv_mass[a] + inv_mass[b];

                if inv_mass_sum <= 0.0 {
                    *output = ContactDelta::default();

                    return;
                }

                /*
                    Normal impulse.

                    Momentum conserving:
                        A = -n * j * invMassA
                        B = +n * j * invMassB
                */

                let impulse = if normal_velocity < 0.0 {
                    -(1.0 + restitution) * normal_velocity / inv_mass_sum
                } else {
                    0.0
                };

                let impulse_x = nx * impulse;

                let impulse_y = ny * impulse;

                /*
                    Baumgarte-style positional correction.

                    This is not an impulse and is not applied
                    to momentum; it corrects geometric overlap.
                */
                let correction = (penetration - slop).max(0.0) * correction_percent / inv_mass_sum;

                *output = ContactDelta {
                    ix: impulse_x,
                    iy: impulse_y,

                    px: nx * correction,

                    py: ny * correction,
                };
            });
    }

    fn reduce_a(&mut self, entities: &EntitiesSoa) {
        let inv_mass = &entities.inv_mass;

        let deltas = &self.contact_deltas;

        let by_a = &self.by_a;

        let offsets = &self.a_offsets;

        self.delta_velocity_x[..entities.len()]
            .par_iter_mut()
            .zip(self.delta_velocity_y[..entities.len()].par_iter_mut())
            .zip(self.delta_position_x[..entities.len()].par_iter_mut())
            .zip(self.delta_position_y[..entities.len()].par_iter_mut())
            .enumerate()
            .for_each(|(entity, (((dvx, dvy), dpx), dpy))| {
                let start = offsets[entity];

                let end = offsets[entity + 1];

                let mut ix = 0.0;
                let mut iy = 0.0;

                let mut px = 0.0;
                let mut py = 0.0;

                for index in start..end {
                    let contact = by_a[index] as usize;

                    let delta = deltas[contact];

                    ix += delta.ix;
                    iy += delta.iy;

                    px += delta.px;
                    py += delta.py;
                }

                let im = inv_mass[entity];

                /*
                    A receives the negative impulse
                    and negative position correction.
                */
                *dvx = -ix * im;

                *dvy = -iy * im;

                *dpx = -px * im;

                *dpy = -py * im;
            });
    }

    fn reduce_b(&mut self, entities: &EntitiesSoa) {
        let inv_mass = &entities.inv_mass;

        let deltas = &self.contact_deltas;

        let by_b = &self.by_b;

        let offsets = &self.b_offsets;

        self.delta_velocity_x[..entities.len()]
            .par_iter_mut()
            .zip(self.delta_velocity_y[..entities.len()].par_iter_mut())
            .zip(self.delta_position_x[..entities.len()].par_iter_mut())
            .zip(self.delta_position_y[..entities.len()].par_iter_mut())
            .enumerate()
            .for_each(|(entity, (((dvx, dvy), dpx), dpy))| {
                let start = offsets[entity];

                let end = offsets[entity + 1];

                let mut ix = 0.0;
                let mut iy = 0.0;

                let mut px = 0.0;
                let mut py = 0.0;

                for index in start..end {
                    let contact = by_b[index] as usize;

                    let delta = deltas[contact];

                    ix += delta.ix;
                    iy += delta.iy;

                    px += delta.px;
                    py += delta.py;
                }

                let im = inv_mass[entity];

                /*
                    B receives positive impulse
                    and positive correction.
                */
                *dvx += ix * im;

                *dvy += iy * im;

                *dpx += px * im;

                *dpy += py * im;
            });
    }
}
