// packedmatrix::storage.rs

/// Read-only contiguous packed storage.
pub trait PackedStorage<T> {
    fn as_slice(&self) -> &[T];
}

/// Mutable contiguous packed storage.
pub trait PackedStorageMut<T>: PackedStorage<T> {
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