use crate::entities::entity_id::EntityId;

pub struct EntityStorage {
    pub id: Vec<EntityId>,

    pub position_x: Vec<f32>,
    pub position_y: Vec<f32>,

    pub velocity_x: Vec<f32>,
    pub velocity_y: Vec<f32>,

    pub facing_x: Vec<f32>,
    pub facing_y: Vec<f32>,

    pub force_x: Vec<f32>,
    pub force_y: Vec<f32>,

    pub mass: Vec<f32>,
    pub inv_mass: Vec<f32>,

    pub size: Vec<f32>,
    pub sensing_radius: Vec<f32>,
}

impl EntityStorage {
    pub fn new(max_storage: usize) -> Self {
        Self {
            id: Vec::with_capacity(max_storage),

            position_x: Vec::with_capacity(max_storage),
            position_y: Vec::with_capacity(max_storage),

            velocity_x: Vec::with_capacity(max_storage),
            velocity_y: Vec::with_capacity(max_storage),

            facing_x: Vec::with_capacity(max_storage),
            facing_y: Vec::with_capacity(max_storage),

            force_x: Vec::with_capacity(max_storage),
            force_y: Vec::with_capacity(max_storage),

            mass: Vec::with_capacity(max_storage),
            inv_mass: Vec::with_capacity(max_storage),

            size: Vec::with_capacity(max_storage),
            sensing_radius: Vec::with_capacity(max_storage),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.id.len()
    }

    pub fn insert(
        &mut self,
        id: EntityId,
        position: (f32, f32),
        velocity: (f32, f32),
        facing: Option<(f32, f32)>,
        force: Option<(f32, f32)>,
        mass: f32,
        size: f32,
        sensing_radius: f32,
    ) -> usize {
        let c_idx = self.len();

        self.id.push(id);

        self.position_x.push(position.0);
        self.position_y.push(position.1);

        self.velocity_x.push(velocity.0);
        self.velocity_y.push(velocity.1);

        if let Some((x, y)) = facing {
            self.facing_x.push(x);
            self.facing_y.push(y);
        } else {
            self.facing_x.push(0.0);
            self.facing_y.push(0.0);
        }

        if let Some((x, y)) = force {
            self.force_x.push(x);
            self.force_y.push(y);
        } else {
            self.force_x.push(0.0);
            self.force_y.push(0.0);
        }

        self.mass.push(mass);
        if mass > 0.0 {
            self.inv_mass.push(1.0 / mass);
        } else {
            self.inv_mass.push(0.0);
        }

        self.size.push(size);
        self.sensing_radius.push(sensing_radius);

        c_idx
    }

    pub fn remove(&mut self, idx: usize) -> (Option<EntityId>, Option<EntityId>) {
        let last = self.len() - 1;

        let swapped_id = if idx < last {
            self.id.get(last).copied()
        } else {
            None
        };

        if idx < last {
            self.id.swap(idx, last);

            self.position_x.swap(idx, last);
            self.position_y.swap(idx, last);

            self.velocity_x.swap(idx, last);
            self.velocity_y.swap(idx, last);

            self.facing_x.swap(idx, last);
            self.facing_y.swap(idx, last);

            self.force_x.swap(idx, last);
            self.force_y.swap(idx, last);

            self.mass.swap(idx, last);
            self.inv_mass.swap(idx, last);

            self.size.swap(idx, last);
            self.sensing_radius.swap(idx, last);
        }

        let removed_id = self.id.pop();

        self.position_x.pop();
        self.position_y.pop();

        self.velocity_x.pop();
        self.velocity_y.pop();

        self.facing_x.pop();
        self.facing_y.pop();

        self.force_x.pop();
        self.force_y.pop();

        self.mass.pop();
        self.inv_mass.pop();

        self.size.pop();
        self.sensing_radius.pop();

        (removed_id, swapped_id)
    }
}
