//! Contiguous storage abstractions behind owned matrices and borrowed views.
//!
//! [`PackedStorage`] supports read-only packed operations; [`PackedStorageMut`]
//! additionally enables mutation and destructive LAPACK calls. `Vec<T>`, `&[T]`,
//! and `&mut [T]` provide the standard owned, view, and mutable-view forms. Most
//! users construct a matrix with `from_vec`, `from_slice`, or `from_slice_mut`
//! instead of naming these traits directly. See the [crate guide](crate) for the
//! ownership and allocation policy.

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
