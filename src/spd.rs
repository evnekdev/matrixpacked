// packedmatrix::spd.rs
//! Symmetric/Hermitian positive-definite-intended packed matrices.
//!
//! [`PackedSPD`] stores the lower triangle and supports real SPD or complex HPD
//! workflows. Basic constructors record intent but do not prove positive
//! definiteness; Cholesky and strict nalgebra conversion validate it. A common
//! workflow computes [`crate::PackedCholesky`], reuses it for solves and
//! refinement, and inspects condition/determinant diagnostics. Convert explicitly
//! to [`crate::symmetric`] or [`crate::hermitian`] before unrestricted updates.

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

/// A positive-definite-intended matrix using lower packed storage.
///
/// Real scalars represent symmetric positive-definite (SPD) matrices. Complex
/// scalars represent Hermitian positive-definite (HPD) matrices and mirrored
/// reads are conjugated. Physical lower columns contain `n * (n + 1) / 2`
/// elements.
///
/// This type records intent: ordinary constructors and mutable access do **not**
/// prove or preserve positive definiteness. Cholesky-backed operations report an
/// error if the stored matrix is not positive definite. With `nalgebra-interop`,
/// strict constructors validate structure and positive definiteness.
///
/// # Examples
///
/// ```
/// use matrixpacked::PackedSPD;
/// let matrix = PackedSPD::from_vec(2, vec![4.0_f64, 1.0, 3.0])?;
/// assert_eq!(matrix.get(0, 1)?, 1.0);
/// // Construction validates layout, while Cholesky performs the numerical test.
/// # Ok::<(), matrixpacked::PackedMatrixError>(())
/// ```
#[derive(Clone)]
pub struct PackedSPD<T, S = Vec<T>> {
    n: usize,
    data: S,
    marker: PhantomData<T>,
}

/// Immutable packed lower-triangular matrix view.
pub type PackedSPDView<'a, T> = PackedSPD<T, &'a [T]>;
/// Mutable packed lower-triangular matrix view.
pub type PackedSPDViewMut<'a, T> = PackedSPD<T, &'a mut [T]>;

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedSPD<T, S> {
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

impl<S> PackedSPD<f32, S> {
    /// Consumes the SPD wrapper and returns the less restrictive symmetric matrix.
    /// This is the intended route before applying an unrestricted rank update.
    pub fn into_symmetric(self) -> crate::PackedSymmetric<f32, S> {
        crate::PackedSymmetric::from_storage(self.n, self.data)
    }
}

impl<S> PackedSPD<f64, S> {
    /// Consumes the SPD wrapper and returns the less restrictive symmetric matrix.
    /// This is the intended route before applying an unrestricted rank update.
    pub fn into_symmetric(self) -> crate::PackedSymmetric<f64, S> {
        crate::PackedSymmetric::from_storage(self.n, self.data)
    }
}

impl<S> PackedSPD<num_complex::Complex32, S> {
    /// Consumes the HPD wrapper and returns the less restrictive Hermitian matrix.
    /// This is the intended route before applying an unrestricted rank update.
    pub fn into_hermitian(self) -> crate::PackedHermitian<num_complex::Complex32, S> {
        crate::PackedHermitian::from_storage(self.n, self.data)
    }
}

impl<S> PackedSPD<num_complex::Complex64, S> {
    /// Consumes the HPD wrapper and returns the less restrictive Hermitian matrix.
    /// This is the intended route before applying an unrestricted rank update.
    pub fn into_hermitian(self) -> crate::PackedHermitian<num_complex::Complex64, S> {
        crate::PackedHermitian::from_storage(self.n, self.data)
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedSPD<T, S> {
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
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedSPD<T, S>
where
    S: PackedStorage<T>,
{
    /// Borrows the stored lower triangle in packed-column order.
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Returns a reference only if the element is physically stored.
    ///
    /// Mirrored upper-triangle coordinates return `None`; use `get` for logical access.
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
    pub fn as_view(&self) -> PackedSPDView<'_, T> {
        PackedSPD {
            n: self.n,
            data: self.as_slice(),
            marker: PhantomData,
        }
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedSPD<T, S>
where
    T: LapackScalar,
    S: PackedStorage<T>,
{
    /// Returns the logical symmetric matrix value.
    pub fn get(&self, row: usize, col: usize) -> Result<T, PackedMatrixError> {
        if !self.contains_index(row, col) {
            return Err(PackedMatrixError::IndexOutOfBounds {
                row,
                col,
                n: self.n,
            });
        }
        let value = *self
            .as_slice()
            .get(self.packed_index(row, col).expect("valid packed index"))
            .expect("valid packed index");
        Ok(if row >= col { value } else { value.conjugate() })
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T, S> PackedSPD<T, S>
where
    S: PackedStorageMut<T>,
{
    /// Mutably borrows the stored lower triangle in packed-column order.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// Returns mutable access only for a physically stored lower-triangle coordinate.
    pub fn get_stored_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if !self.is_stored(row, col) {
            return None;
        }
        let index = self.packed_index(row, col)?;
        self.as_mut_slice().get_mut(index)
    }

    /// Returns mutable access to a stored lower-triangle element.
    ///
    /// # Errors
    ///
    /// Returns an out-of-bounds error or [`PackedMatrixError::StructuralZero`]
    /// for a mirrored upper coordinate.
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
    /// Mirrored coordinates update the same packed element.
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<(), PackedMatrixError>
    where
        T: LapackScalar,
    {
        if !self.contains_index(row, col) {
            return Err(PackedMatrixError::IndexOutOfBounds {
                row,
                col,
                n: self.n,
            });
        }
        let stored_value = if row >= col { value } else { value.conjugate() };
        let index = self.packed_index(row, col).expect("valid packed index");
        self.as_mut_slice()[index] = stored_value;
        Ok(())
    }
    /// Fill all physically available elements with the same value.
    pub fn fill_stored(&mut self, value: T)
    where
        T: Copy,
    {
        self.as_mut_slice().fill(value);
    }

    /// Creates a mutable view whose changes update this matrix.
    ///
    /// Mutating storage may invalidate the positive-definite intent.
    pub fn as_view_mut(&mut self) -> PackedSPDViewMut<'_, T> {
        let n = self.n;

        PackedSPD {
            n,
            data: self.as_mut_slice(),
            marker: PhantomData,
        }
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

impl<T> PackedSPD<T, Vec<T>> {
    /// Creates an owned positive-definite-intended matrix from lower packed columns.
    ///
    /// This validates storage length only, not symmetry/Hermitian diagonal rules
    /// or positive definiteness.
    ///
    /// # Errors
    ///
    /// Returns an error if the packed length is wrong or overflows.
    pub fn from_vec(n: usize, data: Vec<T>) -> Result<Self, PackedMatrixError> {
        Self::validate_len(n, data.len())?;
        Ok(Self {
            n,
            data,
            marker: PhantomData,
        })
    }

    /// Generates the stored lower triangle in packed-column order.
    ///
    /// # Errors
    ///
    /// Returns an error if the packed length overflows.
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

impl<T> PackedSPD<T, Vec<T>>
where
    T: LapackScalar,
{
    /// Creates an owned all-zero matrix carrying positive-definite intent.
    ///
    /// The zero matrix is not positive definite; this constructor is primarily
    /// useful as writable storage.
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

impl<T> PackedSPD<T, Vec<T>>
where
    T: LapackScalar + One,
{
    /// Creates an owned identity matrix, which is positive definite.
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

impl<'a, T> PackedSPD<T, &'a [T]> {
    /// Creates an immutable positive-definite-intended view without copying.
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

impl<'a, T> PackedSPD<T, &'a mut [T]> {
    /// Creates a mutable positive-definite-intended view without copying.
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

impl<T, S> Index<(usize, usize)> for PackedSPD<T, S>
where
    S: PackedStorage<T>,
{
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.try_get(row, col)
            .unwrap_or_else(|error| panic!("invalid packed spd-matrix indexing: {error}"))
    }
}

impl<T, S> IndexMut<(usize, usize)> for PackedSPD<T, S>
where
    S: PackedStorageMut<T>,
{
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        self.try_get_mut(row, col)
            .unwrap_or_else(|error| panic!("invalid mutable packed spd-matrix indexing: {error}"))
    }
}

/***********************************************************************************************************************************************************************/
/***********************************************************************************************************************************************************************/

/***********************************************************************************************************************************************************************/
/* FORMATTING                                                                                                                                                          */
/***********************************************************************************************************************************************************************/

impl<T, S> std::fmt::Debug for PackedSPD<T, S>
where
    T: LapackScalar,
    S: PackedStorage<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::formatting::debug_square(formatter, self.n, |row, col| {
            self.get(row, col).expect("valid SPD coordinate")
        })
    }
}

impl<T, S> std::fmt::Display for PackedSPD<T, S>
where
    T: LapackScalar + std::fmt::Display,
    S: PackedStorage<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::formatting::display_square(formatter, self.n, |row, col| {
            self.get(row, col).expect("valid SPD coordinate")
        })
    }
}

impl<T, L, R> std::ops::Add<&PackedSPD<T, R>> for &PackedSPD<T, L>
where
    T: LapackScalar,
    L: PackedStorage<T>,
    R: PackedStorage<T>,
{
    type Output = PackedSPD<T>;
    fn add(self, rhs: &PackedSPD<T, R>) -> Self::Output {
        assert_eq!(
            self.dimension(),
            rhs.dimension(),
            "matrix dimensions must match"
        );
        PackedSPD::from_vec(
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
impl<T, S, R> std::ops::AddAssign<&PackedSPD<T, R>> for PackedSPD<T, S>
where
    T: LapackScalar,
    S: PackedStorageMut<T>,
    R: PackedStorage<T>,
{
    fn add_assign(&mut self, rhs: &PackedSPD<T, R>) {
        assert_eq!(
            self.dimension(),
            rhs.dimension(),
            "matrix dimensions must match"
        );
        for (a, &b) in self.as_mut_slice().iter_mut().zip(rhs.as_slice()) {
            *a += b;
        }
    }
}

impl<T, S> PackedSPD<T, S>
where
    T: crate::backend::PositiveDefinitePackedBackend,
    S: PackedStorage<T>,
{
    /// Computes `y = alpha * A * x + beta * y` using packed symmetric/Hermitian BLAS.
    ///
    /// # Errors
    ///
    /// Returns an error unless both vectors have length `n`, or if `n` exceeds
    /// the backend integer range.
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
            T::pmv(
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
    /// Allocates and returns `A * x`.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid vector length or backend dimension.
    pub fn mul_vector(&self, x: &[T]) -> Result<Vec<T>, PackedMatrixError> {
        crate::factorization::check_rhs(self.n, x)?;
        let mut y = vec![T::zero(); self.n];
        self.mul_vector_into(x, &mut y, T::one(), T::zero())?;
        Ok(y)
    }
    /// Copies and factors `A` as `L Lᴴ` using packed Cholesky (`xPPTRF`).
    ///
    /// # Errors
    ///
    /// Returns [`PackedMatrixError::FactorizationFailure`] when `A` is not
    /// positive definite, or an error for invalid dimensions.
    pub fn cholesky(&self) -> Result<crate::factorization::PackedCholesky<T>, PackedMatrixError> {
        crate::factorization::PackedCholesky::factorize_storage(
            self.n,
            self.as_slice().to_vec(),
            b'L',
        )
    }
    /// Solves `A x = b` using a temporary Cholesky factorization.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid RHS length or if Cholesky fails.
    pub fn solve_vector(&self, b: &[T]) -> Result<Vec<T>, PackedMatrixError> {
        self.cholesky()?.solve_vector(b)
    }
    /// Returns an owned packed inverse, leaving this matrix unchanged.
    /// This allocates packed storage, factorizes the copy, and destroys that factorization.
    pub fn inverse(&self) -> Result<PackedSPD<T>, PackedMatrixError> {
        self.cholesky()?.into_inverse()
    }
}
impl<T, S> PackedSPD<T, S>
where
    T: crate::backend::PositiveDefinitePackedBackend,
    S: PackedStorageMut<T>,
{
    /// Consumes mutable packed storage and overwrites it with Cholesky factors.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not positive definite or dimensions
    /// are invalid.
    pub fn cholesky_in_place(
        self,
    ) -> Result<crate::factorization::PackedCholesky<T, S>, PackedMatrixError> {
        crate::factorization::PackedCholesky::factorize_storage(self.n, self.data, b'L')
    }
}
impl<T, S> std::ops::Mul<&[T]> for &PackedSPD<T, S>
where
    T: crate::backend::PositiveDefinitePackedBackend,
    S: PackedStorage<T>,
{
    type Output = Vec<T>;
    fn mul(self, rhs: &[T]) -> Self::Output {
        self.mul_vector(rhs)
            .expect("matrix/vector dimensions must match")
    }
}
