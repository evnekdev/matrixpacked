// packedmatrix::storage.rs

/// Read-only contiguous packed storage.
pub trait PackedStorage<T> {
    /// Borrows the contiguous packed elements in their physical column-major order.
    fn as_slice(&self) -> &[T];
}

/// Mutable contiguous packed storage.
pub trait PackedStorageMut<T>: PackedStorage<T> {
    /// Mutably borrows the contiguous packed elements in their physical order.
    fn as_mut_slice(&mut self) -> &mut [T];
}

impl<T> PackedStorage<T> for Vec<T> {
    fn as_slice(&self) -> &[T] {
        self
    }
}

impl<T> PackedStorageMut<T> for Vec<T> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

impl<T> PackedStorage<T> for &[T] {
    fn as_slice(&self) -> &[T] {
        self
    }
}

impl<T> PackedStorage<T> for &mut [T] {
    fn as_slice(&self) -> &[T] {
        self
    }
}

impl<T> PackedStorageMut<T> for &mut [T] {
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}
