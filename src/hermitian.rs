// packedmatrix::hermitian.rs
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

/// A Hermitian matrix stored using its lower triangle in LAPACK packed-column format.
/// Upper-triangle reads return the conjugate of the corresponding stored lower-triangle element.
#[derive(Clone)]
pub struct PackedHermitian<T, S = Vec<T>> {
    n: usize,
    data: S,
    marker: PhantomData<T>,
}

/// Immutable packed lower-triangular matrix view.
pub type PackedHermitianView<'a, T> = PackedHermitian<T, &'a [T]>;
/// Mutable packed lower-triangular matrix view.
pub type PackedHermitianViewMut<'a, T> = PackedHermitian<T, &'a mut [T]>;

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedHermitian<T, S> {
    pub(crate) fn into_storage(self) -> S {
        self.data
    }
    pub(crate) fn from_storage(n: usize, data: S) -> Self {
        Self {
            n,
            data,
            marker: PhantomData,
        }
    }

    /// Number of packed elements required for an `n x n` matrix.
    pub fn packed_len(n: usize) -> Result<usize, PackedMatrixError> {
        return n
            .checked_add(1)
            .and_then(|n1| n.checked_mul(n1))
            .map(|value| value / 2)
            .ok_or(PackedMatrixError::DimensionOverflow { n });
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

impl<T, S> PackedHermitian<T, S> {
    /// Returns the physical packed index for a stored matrix coordinate.
    ///
    /// Returns `None` for:
    ///
    /// - an out-of-bounds coordinate;
    /// - an mirrored coordinate.
    pub fn packed_index(&self, row: usize, col: usize) -> Option<usize> {
        if !self.contains_index(row, col) {
            return None;
        }
        let (row, col) = if row >= col { (row, col) } else { (col, row) };
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

        Ok(self
            .packed_index(row, col)
            .expect("in-bounds Hermitian index"))
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedHermitian<T, S>
where
    S: PackedStorage<T>,
{
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Returns a reference only if the element is physically stored.
    ///
    /// Upper-triangle coordinates return `None`; use `get` for conjugating logical access.
    pub fn get_stored(&self, row: usize, col: usize) -> Option<&T> {
        if !self.is_stored(row, col) {
            return None;
        }
        let index = self.packed_index(row, col)?;
        self.as_slice().get(index)
    }

    /// Checked access to a physically stored element.
    ///
    /// This returns an error for mirrored coordinates.
    pub fn try_get(&self, row: usize, col: usize) -> Result<&T, PackedMatrixError> {
        if !self.contains_index(row, col) {
            return Err(PackedMatrixError::IndexOutOfBounds {
                row,
                col,
                n: self.n,
            });
        }
        self.get_stored(row, col)
            .ok_or(PackedMatrixError::StructuralZero { row, col })
    }

    /// Creates an immutable view.
    pub fn as_view(&self) -> PackedHermitianView<'_, T> {
        PackedHermitian {
            n: self.n,
            data: self.as_slice(),
            marker: PhantomData,
        }
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedHermitian<T, S>
where
    T: LapackScalar,
    S: PackedStorage<T>,
{
    /// Returns the logical Hermitian value, conjugating mirrored entries.
    pub fn get(&self, row: usize, col: usize) -> Result<T, PackedMatrixError> {
        if !self.contains_index(row, col) {
            return Err(PackedMatrixError::IndexOutOfBounds {
                row,
                col,
                n: self.n,
            });
        }
        let value = *self
            .get_stored(row.max(col), row.min(col))
            .expect("valid packed index");
        Ok(if row >= col { value } else { value.conjugate() })
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedHermitian<T, S>
where
    T: LapackScalar,
    S: PackedStorageMut<T>,
{
    /// TODO
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// TODO
    pub fn get_stored_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if !self.is_stored(row, col) {
            return None;
        }
        let index = self.packed_index(row, col)?;
        self.as_mut_slice().get_mut(index)
    }

    /// TODO
    pub fn try_get_mut(&mut self, row: usize, col: usize) -> Result<&mut T, PackedMatrixError> {
        if !self.contains_index(row, col) {
            return Err(PackedMatrixError::IndexOutOfBounds {
                row,
                col,
                n: self.n,
            });
        }
        self.get_stored_mut(row, col)
            .ok_or(PackedMatrixError::StructuralZero { row, col })
    }

    /// Sets a logical matrix element.
    ///
    /// Upper-triangle values are conjugated before being written to lower storage.
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<(), PackedMatrixError> {
        if !self.contains_index(row, col) {
            return Err(PackedMatrixError::IndexOutOfBounds {
                row,
                col,
                n: self.n,
            });
        }
        let (stored_row, stored_col, stored_value) = if row >= col {
            (row, col, value)
        } else {
            (col, row, value.conjugate())
        };
        *self
            .get_stored_mut(stored_row, stored_col)
            .expect("valid stored coordinate") = stored_value;
        Ok(())
    }
    /// Fill all physically available elements with the same value.
    pub fn fill_stored(&mut self, value: T)
    where
        T: Copy,
    {
        self.as_mut_slice().fill(value);
    }

    pub fn as_view_mut(&mut self) -> PackedHermitianViewMut<'_, T> {
        let n = self.n;

        PackedHermitian {
            n,
            data: self.as_mut_slice(),
            marker: PhantomData,
        }
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T> PackedHermitian<T, Vec<T>> {
    /// TODO
    pub fn from_vec(n: usize, data: Vec<T>) -> Result<Self, PackedMatrixError> {
        Self::validate_len(n, data.len())?;
        return Ok(Self {
            n,
            data,
            marker: PhantomData,
        });
    }

    /// TODO
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
        return Ok(Self {
            n,
            data,
            marker: PhantomData,
        });
    }

    /// Convert into a conventional `Vec<T>`.
    pub fn into_vec(self) -> Vec<T> {
        return self.data;
    }
}

impl<T> PackedHermitian<T, Vec<T>>
where
    T: LapackScalar,
{
    pub fn zeros(n: usize) -> Result<Self, PackedMatrixError> {
        let len = Self::packed_len(n)?;
        return Ok(Self {
            n,
            data: vec![T::zero(); len],
            marker: PhantomData,
        });
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T> PackedHermitian<T, Vec<T>>
where
    T: LapackScalar + One,
{
    pub fn identity(n: usize) -> Result<Self, PackedMatrixError> {
        let mut matrix = Self::zeros(n)?;
        for i in 0..n {
            matrix.set(i, i, T::one())?;
        }
        return Ok(matrix);
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<'a, T> PackedHermitian<T, &'a [T]> {
    pub fn from_slice(n: usize, data: &'a [T]) -> Result<Self, PackedMatrixError> {
        Self::validate_len(n, data.len())?;
        return Ok(Self {
            n,
            data,
            marker: PhantomData,
        });
    }
}

impl<'a, T> PackedHermitian<T, &'a mut [T]> {
    pub fn from_slice_mut(n: usize, data: &'a mut [T]) -> Result<Self, PackedMatrixError> {
        Self::validate_len(n, data.len())?;
        return Ok(Self {
            n,
            data,
            marker: PhantomData,
        });
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> Index<(usize, usize)> for PackedHermitian<T, S>
where
    S: PackedStorage<T>,
{
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        return self
            .try_get(row, col)
            .unwrap_or_else(|error| panic!("invalid packed hermitian-matrix indexing: {error}"));
    }
}

impl<T, S> IndexMut<(usize, usize)> for PackedHermitian<T, S>
where
    T: LapackScalar,
    S: PackedStorageMut<T>,
{
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        return self.try_get_mut(row, col).unwrap_or_else(|error| {
            panic!("invalid mutable packed hermitian-matrix indexing: {error}")
        });
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

/***********************************************************************************************************************************************************************/
/* FORMATTING                                                                                                                                                          */
/***********************************************************************************************************************************************************************/

impl<T, S> std::fmt::Debug for PackedHermitian<T, S>
where
    T: LapackScalar,
    S: PackedStorage<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::formatting::debug_square(formatter, self.n, |row, col| {
            let value = *self
                .get_stored(row.max(col), row.min(col))
                .expect("valid Hermitian coordinate");
            if row >= col { value } else { value.conjugate() }
        })
    }
}

impl<T, S> std::fmt::Display for PackedHermitian<T, S>
where
    T: LapackScalar + std::fmt::Display,
    S: PackedStorage<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::formatting::display_square(formatter, self.n, |row, col| {
            let value = *self
                .get_stored(row.max(col), row.min(col))
                .expect("valid Hermitian coordinate");
            if row >= col { value } else { value.conjugate() }
        })
    }
}

impl<T, L, R> std::ops::Add<&PackedHermitian<T, R>> for &PackedHermitian<T, L>
where
    T: LapackScalar,
    L: PackedStorage<T>,
    R: PackedStorage<T>,
{
    type Output = PackedHermitian<T>;
    fn add(self, rhs: &PackedHermitian<T, R>) -> Self::Output {
        assert_eq!(
            self.dimension(),
            rhs.dimension(),
            "matrix dimensions must match"
        );
        PackedHermitian::from_vec(
            self.dimension(),
            self.as_slice()
                .iter()
                .zip(rhs.as_slice())
                .map(|(&a, &b)| a + b)
                .collect(),
        )
        .expect("validated packed length")
    }
}
impl<T, L, R> std::ops::Sub<&PackedHermitian<T, R>> for &PackedHermitian<T, L>
where
    T: LapackScalar,
    L: PackedStorage<T>,
    R: PackedStorage<T>,
{
    type Output = PackedHermitian<T>;
    fn sub(self, rhs: &PackedHermitian<T, R>) -> Self::Output {
        assert_eq!(
            self.dimension(),
            rhs.dimension(),
            "matrix dimensions must match"
        );
        PackedHermitian::from_vec(
            self.dimension(),
            self.as_slice()
                .iter()
                .zip(rhs.as_slice())
                .map(|(&a, &b)| a - b)
                .collect(),
        )
        .expect("validated packed length")
    }
}
impl<T, S> std::ops::Neg for &PackedHermitian<T, S>
where
    T: LapackScalar,
    S: PackedStorage<T>,
{
    type Output = PackedHermitian<T>;
    fn neg(self) -> Self::Output {
        PackedHermitian::from_vec(
            self.dimension(),
            self.as_slice().iter().map(|&x| -x).collect(),
        )
        .expect("validated packed length")
    }
}

impl<T, S> PackedHermitian<T, S>
where
    T: crate::backend::HermitianPackedBackend,
    S: PackedStorage<T>,
{
    pub fn mul_vector_into(
        &self,
        x: &[T],
        y: &mut [T],
        alpha: T,
        beta: T,
    ) -> Result<(), PackedMatrixError> {
        crate::factorization::check_rhs(self.n, x)?;
        crate::factorization::check_rhs(self.n, y)?;
        unsafe {
            T::hpmv(
                b'L',
                crate::factorization::checked_n(self.n)?,
                alpha,
                self.as_slice(),
                x,
                beta,
                y,
            )
        };
        Ok(())
    }
    pub fn mul_vector(&self, x: &[T]) -> Result<Vec<T>, PackedMatrixError> {
        crate::factorization::check_rhs(self.n, x)?;
        let mut y = vec![T::zero(); self.n];
        self.mul_vector_into(x, &mut y, T::one(), T::zero())?;
        Ok(y)
    }
    pub fn factorize(
        &self,
    ) -> Result<crate::factorization::PackedHermitianFactor<T>, PackedMatrixError> {
        crate::factorization::PackedHermitianFactor::factorize_storage(
            self.n,
            self.as_slice().to_vec(),
            b'L',
        )
    }
    pub fn solve_vector(&self, b: &[T]) -> Result<Vec<T>, PackedMatrixError> {
        self.factorize()?.solve_vector(b)
    }
    /// Returns an owned Hermitian packed inverse after factorizing a packed copy.
    pub fn inverse(&self) -> Result<PackedHermitian<T>, PackedMatrixError> {
        self.factorize()?.into_inverse()
    }
}
impl<T, S> PackedHermitian<T, S>
where
    T: crate::backend::HermitianPackedBackend,
    S: PackedStorageMut<T>,
{
    pub fn factorize_in_place(
        self,
    ) -> Result<crate::factorization::PackedHermitianFactor<T, S>, PackedMatrixError> {
        crate::factorization::PackedHermitianFactor::factorize_storage(self.n, self.data, b'L')
    }
}
impl<T, S> std::ops::Mul<&[T]> for &PackedHermitian<T, S>
where
    T: crate::backend::HermitianPackedBackend,
    S: PackedStorage<T>,
{
    type Output = Vec<T>;
    fn mul(self, rhs: &[T]) -> Self::Output {
        self.mul_vector(rhs)
            .expect("matrix/vector dimensions must match")
    }
}
