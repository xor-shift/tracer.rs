pub struct NBuffered<T: Sized, const N: usize> {
    swap_count: usize,
    values: [T; N],
}

impl<T: Sized, const N: usize> NBuffered<T, N> {
    pub fn new(values: [T; N]) -> NBuffered<T, N> {
        Self {
            swap_count: 0,
            values,
        }
    }

    pub fn swap(&mut self) {
        self.swap_count += 1;
    }

    pub fn get_mut(&mut self) -> &mut T {
        return &mut self.values[self.swap_count % N];
    }

    pub fn get(&self) -> &T {
        return &self.values[self.swap_count % N];
    }
}

pub struct History<T: Sized, const N: usize> {
    total_pushed: usize,
    values: [T; N],
}

impl<T: Sized, const N: usize> History<T, N> {
    pub fn new() -> History<T, N> {
        Self {
            total_pushed: 0,
            values: todo!(),
        }
    }
}
