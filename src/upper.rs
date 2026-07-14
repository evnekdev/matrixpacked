//! Upper-triangular matrices in traditional packed-column storage.
//!
//! [`PackedUpper`] and its view aliases store only coordinates with
//! `row <= column`; logical lower entries are structural zeros. Construct owned
//! or borrowed storage, then use the shared [`crate::triangular`] operations for
//! multiplication, solves, inverse, norms, conditions, and refinement.
//! [`crate::lower`] documents the complementary lower layout.

// packedmatrix::upper.rs
use num_traits::One;

use crate::{
    error::PackedMatrixError,
    scalar::LapackScalar,
    storage::{PackedStorage, PackedStorageMut},
};

use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
};

/// A square upper-triangular matrix stored in LAPACK packed-column format.
///
/// For an `n x n` matrix, the physical storage is:
///
/// ```text
/// column 0: a00
/// column 1: a01, a11
/// column 2: a02, a12, a22
/// ...
/// ```
///
/// Only coordinates satisfying `row <= col` are physically stored.
///
/// # Examples
///
/// ```
/// use matrixpacked::PackedUpper;
///
/// // Columns are [a00], [a01, a11], [a02, a12, a22].
/// let matrix = PackedUpper::from_vec(3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// assert_eq!(matrix[(1, 2)], 5.0);
/// assert_eq!(matrix.get(2, 0)?, 0.0); // structural lower-triangle zero
/// # Ok::<(), matrixpacked::PackedMatrixError>(())
/// ```
#[derive(Clone)]
pub struct PackedUpper<T, S = Vec<T>> {
    n: usize,
    data: S,
    marker: PhantomData<T>,
}

/// Immutable packed upper-triangular matrix view.
pub type PackedUpperView<'a, T> = PackedUpper<T, &'a [T]>;
/// Mutable packed upper-triangular matrix view.
pub type PackedUpperViewMut<'a, T> = PackedUpper<T, &'a mut [T]>;

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedUpper<T, S> {
    /// Number of packed elements required for an `n x n` matrix.
    pub fn packed_len(n: usize) -> Result<usize, PackedMatrixError> {
        n.checked_add(1)
            .and_then(|n1| n.checked_mul(n1))
            .map(|value| value / 2)
            .ok_or(PackedMatrixError::DimensionOverflow { n })
    }

    fn validate_len(n: usize, actual: usize) -> Result<(), PackedMatrixError> {
        let expected = Self::packed_len(n)?;

        if actual == expected {
            Ok(())
        } else {
            Err(PackedMatrixError::InvalidLength {
                n,
                expected,
                actual,
            })
        }
    }

    /// Number of rows.
    pub const fn nrows(&self) -> usize {
        self.n
    }

    /// Number of columns.
    pub const fn ncols(&self) -> usize {
        self.n
    }

    /// Dimension size (the same as number of columns or rows).
    pub const fn dimension(&self) -> usize {
        self.n
    }
    /// Shape tuple.
    pub fn shape(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    /// Returns whether `(row, col)` lies within the logical matrix.
    pub fn contains_index(&self, row: usize, col: usize) -> bool {
        row < self.n && col < self.n
    }

    /// Returns whether `(row, col)` is physically stored.
    pub fn is_stored(&self, row: usize, col: usize) -> bool {
        self.contains_index(row, col) && row <= col
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedUpper<T, S> {
    /// Returns the physical packed index for a stored matrix coordinate.
    ///
    /// Returns `None` for:
    ///
    /// - an out-of-bounds coordinate;
    /// - an lower-triangular structural zero.
    pub fn packed_index(&self, row: usize, col: usize) -> Option<usize> {
        if !self.is_stored(row, col) {
            return None;
        }

        // Upper-packed columns have lengths 1, 2, ..., n.
        Some(col * (col + 1) / 2 + row)
    }

    fn checked_packed_index(&self, row: usize, col: usize) -> Result<usize, PackedMatrixError> {
        if !self.contains_index(row, col) {
            return Err(PackedMatrixError::IndexOutOfBounds {
                row,
                col,
                n: self.n,
            });
        }

        self.packed_index(row, col)
            .ok_or(PackedMatrixError::StructuralZero { row, col })
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedUpper<T, S>
where
    S: PackedStorage<T>,
{
    /// Borrows the packed elements in upper-column order.
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Returns a reference only if the element is physically stored.
    ///
    /// Upper-triangular structural zeros return `None`.
    pub fn get_stored(&self, row: usize, col: usize) -> Option<&T> {
        let index = self.packed_index(row, col)?;
        self.as_slice().get(index)
    }

    /// Checked access to a physically stored element.
    ///
    /// This returns an error for lower-triangular structural zeros.
    pub fn try_get(&self, row: usize, col: usize) -> Result<&T, PackedMatrixError> {
        let index = self.checked_packed_index(row, col);
        index.map(|index| &self.as_slice()[index])
    }

    /// Creates an immutable view.
    pub fn as_view(&self) -> PackedUpperView<'_, T> {
        PackedUpper {
            n: self.n,
            data: self.as_slice(),
            marker: PhantomData,
        }
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedUpper<T, S>
where
    T: LapackScalar,
    S: PackedStorage<T>,
{
    /// Returns the logical matrix value.
    ///
    /// - Upper triangle: returns the stored value.
    /// - Lower triangle: returns zero.
    /// - Out of bounds: returns an error.
    pub fn get(&self, row: usize, col: usize) -> Result<T, PackedMatrixError> {
        if !self.contains_index(row, col) {
            return Err(PackedMatrixError::IndexOutOfBounds {
                row,
                col,
                n: self.n,
            });
        }

        Ok(match self.get_stored(row, col) {
            Some(value) => *value,
            None => T::zero(),
        })
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedUpper<T, S>
where
    S: PackedStorageMut<T>,
{
    /// Mutably borrows the packed elements in upper-column order.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// Returns a mutable reference only when the zero-based coordinate is stored.
    pub fn get_stored_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        let index = self.packed_index(row, col)?;
        self.as_mut_slice().get_mut(index)
    }

    /// Returns mutable access to a stored element.
    ///
    /// # Errors
    ///
    /// Returns [`PackedMatrixError::IndexOutOfBounds`] outside the matrix or
    /// [`PackedMatrixError::StructuralZero`] for the implicit lower triangle.
    pub fn try_get_mut(&mut self, row: usize, col: usize) -> Result<&mut T, PackedMatrixError> {
        let index = self.checked_packed_index(row, col)?;
        Ok(&mut self.as_mut_slice()[index])
    }

    /// Sets a physically stored upper-triangular element.
    ///
    /// Attempting to set an lower-triangular structural zero is an error.
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<(), PackedMatrixError> {
        *self.try_get_mut(row, col)? = value;
        Ok(())
    }
    /// Fill all physically available elements with the same value.
    pub fn fill_stored(&mut self, value: T)
    where
        T: Copy,
    {
        self.as_mut_slice().fill(value);
    }

    /// Creates a mutable packed view borrowing this matrix's storage.
    pub fn as_view_mut(&mut self) -> PackedUpperViewMut<'_, T> {
        let n = self.n;

        PackedUpper {
            n,
            data: self.as_mut_slice(),
            marker: PhantomData,
        }
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T> PackedUpper<T, Vec<T>> {
    /// Creates an owned matrix from upper-packed column-major data.
    ///
    /// # Errors
    ///
    /// Returns an error unless `data.len() == n * (n + 1) / 2`, or if the
    /// packed length overflows `usize`.
    pub fn from_vec(n: usize, data: Vec<T>) -> Result<Self, PackedMatrixError> {
        Self::validate_len(n, data.len())?;
        Ok(Self {
            n,
            data,
            marker: PhantomData,
        })
    }

    /// Generates stored elements by calling `function(row, column)` in packed order.
    ///
    /// The function is never called for structural lower-triangle zeros.
    ///
    /// # Errors
    ///
    /// Returns [`PackedMatrixError::DimensionOverflow`] if the packed length
    /// cannot be represented.
    pub fn from_fn(
        n: usize,
        mut function: impl FnMut(usize, usize) -> T,
    ) -> Result<Self, PackedMatrixError> {
        let len = Self::packed_len(n)?;
        let mut data = Vec::with_capacity(len);
        // LAPACK upper-packed column order.
        for col in 0..n {
            for row in 0..=col {
                data.push(function(row, col));
            }
        }
        Ok(Self {
            n,
            data,
            marker: PhantomData,
        })
    }

    /// Convert into a conventional `Vec<T>`.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T> PackedUpper<T, Vec<T>>
where
    T: LapackScalar,
{
    /// Creates an owned upper-triangular matrix whose stored elements are zero.
    ///
    /// # Errors
    ///
    /// Returns an error if the packed length overflows.
    pub fn zeros(n: usize) -> Result<Self, PackedMatrixError> {
        let len = Self::packed_len(n)?;
        Ok(Self {
            n,
            data: vec![T::zero(); len],
            marker: PhantomData,
        })
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T> PackedUpper<T, Vec<T>>
where
    T: LapackScalar + One,
{
    /// Creates an owned upper-triangular identity matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the packed length overflows.
    pub fn identity(n: usize) -> Result<Self, PackedMatrixError> {
        let mut matrix = Self::zeros(n)?;
        for i in 0..n {
            matrix.set(i, i, T::one())?;
        }
        Ok(matrix)
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<'a, T> PackedUpper<T, &'a [T]> {
    /// Creates an immutable view over upper-packed column-major data.
    ///
    /// # Errors
    ///
    /// Returns an error when the slice length does not match `n`.
    pub fn from_slice(n: usize, data: &'a [T]) -> Result<Self, PackedMatrixError> {
        Self::validate_len(n, data.len())?;
        Ok(Self {
            n,
            data,
            marker: PhantomData,
        })
    }
}

impl<'a, T> PackedUpper<T, &'a mut [T]> {
    /// Creates a mutable view over upper-packed column-major data.
    ///
    /// Mutations through the view update the caller's slice.
    ///
    /// # Errors
    ///
    /// Returns an error when the slice length does not match `n`.
    pub fn from_slice_mut(n: usize, data: &'a mut [T]) -> Result<Self, PackedMatrixError> {
        Self::validate_len(n, data.len())?;
        Ok(Self {
            n,
            data,
            marker: PhantomData,
        })
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> Index<(usize, usize)> for PackedUpper<T, S>
where
    S: PackedStorage<T>,
{
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.try_get(row, col)
            .unwrap_or_else(|error| panic!("invalid packed upper-matrix indexing: {error}"))
    }
}

impl<T, S> IndexMut<(usize, usize)> for PackedUpper<T, S>
where
    S: PackedStorageMut<T>,
{
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        self.try_get_mut(row, col)
            .unwrap_or_else(|error| panic!("invalid mutable packed upper-matrix indexing: {error}"))
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

/***********************************************************************************************************************************************************************/
/* FORMATTING                                                                                                                                                          */
/***********************************************************************************************************************************************************************/

impl<T, S> std::fmt::Debug for PackedUpper<T, S>
where
    T: LapackScalar,
    S: PackedStorage<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::formatting::debug_square(formatter, self.n, |row, col| {
            if row <= col {
                *self.get_stored(row, col).expect("valid stored coordinate")
            } else {
                T::zero()
            }
        })
    }
}

impl<T, S> std::fmt::Display for PackedUpper<T, S>
where
    T: LapackScalar + std::fmt::Display,
    S: PackedStorage<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::formatting::display_square(formatter, self.n, |row, col| {
            if row <= col {
                *self.get_stored(row, col).expect("valid stored coordinate")
            } else {
                T::zero()
            }
        })
    }
}

crate::arithmetic::impl_packed_ring_ops!(PackedUpper);

crate::triangular::impl_triangular_packed_ops!(PackedUpper, b'U', "upper-triangular");
impl<T, S> std::ops::Mul<&[T]> for &PackedUpper<T, S>
where
    T: crate::backend::TriangularPackedBackend,
    S: PackedStorage<T>,
{
    type Output = Vec<T>;
    fn mul(self, rhs: &[T]) -> Self::Output {
        self.mul_vector(rhs)
            .expect("matrix/vector dimensions must match")
    }
}
