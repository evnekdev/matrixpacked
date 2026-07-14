// packedmatrix::lower.rs
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

/// A square lower-triangular matrix stored in LAPACK packed-column format.
///
/// For an `n x n` matrix, the physical storage is:
///
/// ```text
/// column 0: a00, a10, a20, ..., a(n-1)0
/// column 1:      a11, a21, ..., a(n-1)1
/// column 2:           a22, ..., a(n-1)2
/// ...
/// ```
///
/// Only coordinates satisfying `row >= col` are physically stored.
///
/// # Examples
///
/// ```
/// use matrixpacked::PackedLower;
///
/// // Columns are [a00, a10, a20], [a11, a21], [a22].
/// let mut matrix = PackedLower::from_vec(3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// assert_eq!(matrix[(2, 1)], 5.0);
/// assert_eq!(matrix.get(0, 2)?, 0.0); // structural upper-triangle zero
/// matrix[(2, 1)] = 9.0;
/// assert_eq!(matrix.as_slice(), &[1.0, 2.0, 3.0, 4.0, 9.0, 6.0]);
/// # Ok::<(), matrixpacked::PackedMatrixError>(())
/// ```
#[derive(Clone)]
pub struct PackedLower<T, S = Vec<T>> {
    n: usize,
    data: S,
    marker: PhantomData<T>,
}

/// Immutable packed lower-triangular matrix view.
pub type PackedLowerView<'a, T> = PackedLower<T, &'a [T]>;
/// Mutable packed lower-triangular matrix view.
///
/// Mutations write through to the borrowed packed slice.
///
/// # Examples
///
/// ```
/// use matrixpacked::PackedLower;
///
/// let mut storage = [1, 2, 3];
/// {
///     let mut view = PackedLower::from_slice_mut(2, &mut storage)?;
///     view[(1, 0)] = 7;
/// }
/// assert_eq!(storage, [1, 7, 3]);
/// # Ok::<(), matrixpacked::PackedMatrixError>(())
/// ```
pub type PackedLowerViewMut<'a, T> = PackedLower<T, &'a mut [T]>;

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedLower<T, S> {
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
        self.contains_index(row, col) && row >= col
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedLower<T, S> {
    /// Returns the physical packed index for a stored matrix coordinate.
    ///
    /// Returns `None` for:
    ///
    /// - an out-of-bounds coordinate;
    /// - an upper-triangular structural zero.
    pub fn packed_index(&self, row: usize, col: usize) -> Option<usize> {
        if !self.is_stored(row, col) {
            return None;
        }

        // Safe because dimensions were validated during construction.
        let column_start = col * (2 * self.n - col + 1) / 2;

        Some(column_start + row - col)
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

impl<T, S> PackedLower<T, S>
where
    S: PackedStorage<T>,
{
    /// Borrows the packed elements in lower-column order.
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
    /// This returns an error for upper-triangular structural zeros.
    pub fn try_get(&self, row: usize, col: usize) -> Result<&T, PackedMatrixError> {
        let index = self.checked_packed_index(row, col);
        index.map(|index| &self.as_slice()[index])
    }

    /// Creates an immutable view.
    pub fn as_view(&self) -> PackedLowerView<'_, T> {
        PackedLower {
            n: self.n,
            data: self.as_slice(),
            marker: PhantomData,
        }
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedLower<T, S>
where
    T: LapackScalar,
    S: PackedStorage<T>,
{
    /// Returns the logical matrix value.
    ///
    /// - Lower triangle: returns the stored value.
    /// - Upper triangle: returns zero.
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

impl<T, S> PackedLower<T, S>
where
    S: PackedStorageMut<T>,
{
    /// Mutably borrows the packed elements in lower-column order.
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
    /// [`PackedMatrixError::StructuralZero`] for the implicit upper triangle.
    pub fn try_get_mut(&mut self, row: usize, col: usize) -> Result<&mut T, PackedMatrixError> {
        let index = self.checked_packed_index(row, col)?;
        Ok(&mut self.as_mut_slice()[index])
    }

    /// Sets a physically stored lower-triangular element.
    ///
    /// Attempting to set an upper-triangular structural zero is an error.
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
    pub fn as_view_mut(&mut self) -> PackedLowerViewMut<'_, T> {
        let n = self.n;

        PackedLower {
            n,
            data: self.as_mut_slice(),
            marker: PhantomData,
        }
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T> PackedLower<T, Vec<T>> {
    /// Creates an owned matrix from lower-packed column-major data.
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
    /// The function is never called for structural upper-triangle zeros.
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
        // LAPACK lower-packed column order.
        for col in 0..n {
            for row in col..n {
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

impl<T> PackedLower<T, Vec<T>>
where
    T: LapackScalar,
{
    /// Creates an owned lower-triangular matrix whose stored elements are zero.
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

impl<T> PackedLower<T, Vec<T>>
where
    T: LapackScalar + One,
{
    /// Creates an owned lower-triangular identity matrix.
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

impl<'a, T> PackedLower<T, &'a [T]> {
    /// Creates an immutable view over lower-packed column-major data.
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

impl<'a, T> PackedLower<T, &'a mut [T]> {
    /// Creates a mutable view over lower-packed column-major data.
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

impl<T, S> Index<(usize, usize)> for PackedLower<T, S>
where
    S: PackedStorage<T>,
{
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.try_get(row, col)
            .unwrap_or_else(|error| panic!("invalid packed lower-matrix indexing: {error}"))
    }
}

impl<T, S> IndexMut<(usize, usize)> for PackedLower<T, S>
where
    S: PackedStorageMut<T>,
{
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        self.try_get_mut(row, col)
            .unwrap_or_else(|error| panic!("invalid mutable packed lower-matrix indexing: {error}"))
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

/***********************************************************************************************************************************************************************/
/* FORMATTING                                                                                                                                                          */
/***********************************************************************************************************************************************************************/

impl<T, S> std::fmt::Debug for PackedLower<T, S>
where
    T: LapackScalar,
    S: PackedStorage<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::formatting::debug_square(formatter, self.n, |row, col| {
            if row >= col {
                *self.get_stored(row, col).expect("valid stored coordinate")
            } else {
                T::zero()
            }
        })
    }
}

impl<T, S> std::fmt::Display for PackedLower<T, S>
where
    T: LapackScalar + std::fmt::Display,
    S: PackedStorage<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::formatting::display_square(formatter, self.n, |row, col| {
            if row >= col {
                *self.get_stored(row, col).expect("valid stored coordinate")
            } else {
                T::zero()
            }
        })
    }
}

crate::arithmetic::impl_packed_ring_ops!(PackedLower);

crate::triangular::impl_triangular_packed_ops!(PackedLower, b'L', "lower-triangular");
impl<T, S> std::ops::Mul<&[T]> for &PackedLower<T, S>
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
