use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};
use wide::f32x16;

const LANES: usize = 16;

pub struct EntitiesSoa {
    position_x: Vec<f32>,
    position_y: Vec<f32>,

    velocity_x: Vec<f32>,
    velocity_y: Vec<f32>,

    force_x: Vec<f32>,
    force_y: Vec<f32>,

    mass: Vec<f32>,
    inv_mass: Vec<f32>,
}

impl EntitiesSoa {
    pub fn new(capacity: usize) -> Self {
        Self {
            position_x: Vec::with_capacity(capacity),
            position_y: Vec::with_capacity(capacity),

            velocity_x: Vec::with_capacity(capacity),
            velocity_y: Vec::with_capacity(capacity),

            force_x: Vec::with_capacity(capacity),
            force_y: Vec::with_capacity(capacity),

            mass: Vec::with_capacity(capacity),
            inv_mass: Vec::with_capacity(capacity),
        }
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
    ) -> usize {
        let loc = self.position_x.len();

        self.position_x.push(position_x);
        self.position_y.push(position_y);

        self.velocity_x.push(velocity_x);
        self.velocity_y.push(velocity_y);

        self.force_x.push(force_x);
        self.force_y.push(force_y);

        self.mass.push(mass);
        self.inv_mass.push(1.0 / mass);

        loc
    }

    // pub fn remove(&mut self, index: usize) -> bool {
    //     let len = self.position_x.len();

    //     if index >= len {
    //         return false;
    //     }

    //     let last = len - 1;

    //     if index != last {
    //         self.position_x.swap(index, last);
    //         self.position_y.swap(index, last);

    //         self.velocity_x.swap(index, last);
    //         self.velocity_y.swap(index, last);

    //         self.force_x.swap(index, last);
    //         self.force_y.swap(index, last);

    //         self.mass.swap(index, last);
    //         self.inv_mass.swap(index, last);
    //     }

    //     self.position_x.pop();
    //     self.position_y.pop();

    //     self.velocity_x.pop();
    //     self.velocity_y.pop();

    //     self.force_x.pop();
    //     self.force_y.pop();

    //     self.mass.pop();

    //     true
    // }

    pub fn update_velocities(&mut self, dt: f32) {
        let dt_simd = f32x16::splat(dt);

        let remainder = self.position_x.len() % LANES;
        let simd_len = self.position_x.len() - remainder;

        let (vx_simd, vx_tail) = self.velocity_x.split_at_mut(simd_len);
        let (vy_simd, vy_tail) = self.velocity_y.split_at_mut(simd_len);

        let (fx_simd, fx_tail) = self.force_x.split_at_mut(simd_len);
        let (fy_simd, fy_tail) = self.force_y.split_at_mut(simd_len);

        let (im_simd, im_tail) = self.inv_mass.split_at(simd_len);

        vx_simd
            .par_chunks_exact_mut(LANES)
            .zip(vy_simd.par_chunks_exact_mut(LANES))
            .zip(fx_simd.par_chunks_exact_mut(LANES))
            .zip(fy_simd.par_chunks_exact_mut(LANES))
            .zip(im_simd.par_chunks_exact(LANES))
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

        let remainder = self.position_x.len() % LANES;
        let simd_len = self.position_x.len() - remainder;

        let (px_simd, px_tail) = self.position_x.split_at_mut(simd_len);
        let (py_simd, py_tail) = self.position_y.split_at_mut(simd_len);

        let (vx_simd, vx_tail) = self.velocity_x.split_at(simd_len);
        let (vy_simd, vy_tail) = self.velocity_y.split_at(simd_len);

        px_simd
            .par_chunks_exact_mut(LANES)
            .zip(py_simd.par_chunks_exact_mut(LANES))
            .zip(vx_simd.par_chunks_exact(LANES))
            .zip(vy_simd.par_chunks_exact(LANES))
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
