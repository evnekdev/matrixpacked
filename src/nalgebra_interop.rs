//! Optional conversions between crate-owned full triangular storage and nalgebra.

use crate::{
    FullTriangular, PackedHermitian, PackedLower, PackedMatrixError, PackedSPD, PackedSymmetric,
    PackedUpper, Triangle, backend::PackedFormatConversion, storage::PackedStorage,
};
use nalgebra::{DMatrix, Scalar};
use num_complex::{Complex32, Complex64};
use num_traits::Zero;

impl<T> FullTriangular<T>
where
    T: Scalar,
{
    /// Clones this full column-major buffer into a nalgebra dynamic matrix.
    ///
    /// The result allocates `n * n` entries. This is an owned conversion, not a
    /// zero-copy view; the source remains available afterward.
    pub fn to_dmatrix(&self) -> DMatrix<T> {
        DMatrix::from_vec(self.dimension(), self.dimension(), self.as_slice().to_vec())
    }

    /// Moves this full column-major buffer into a nalgebra dynamic matrix.
    ///
    /// Unlike [`Self::to_dmatrix`], this reuses the owned allocation without
    /// cloning its entries.
    pub fn into_dmatrix(self) -> DMatrix<T> {
        let dimension = self.dimension();
        DMatrix::from_vec(dimension, dimension, self.into_vec())
    }
}

impl<T> FullTriangular<T>
where
    T: Scalar + Zero,
{
    /// Copies a square nalgebra matrix into full triangular storage.
    ///
    /// Nalgebra and `FullTriangular` both use column-major storage, so the full
    /// buffer is copied directly. Entries outside `triangle` are then replaced
    /// by structural zeros to preserve the [`FullTriangular`] invariant.
    /// The conversion allocates `n * n` entries and does not create a view.
    pub fn try_from_dmatrix(
        matrix: &DMatrix<T>,
        triangle: Triangle,
    ) -> Result<Self, PackedMatrixError> {
        let (rows, columns) = matrix.shape();
        if rows != columns {
            return Err(PackedMatrixError::NonSquareMatrix { rows, columns });
        }

        let mut data = matrix.as_slice().to_vec();
        for column in 0..columns {
            for row in 0..rows {
                let is_structural_zero = match triangle {
                    Triangle::Lower => row < column,
                    Triangle::Upper => row > column,
                };
                if is_structural_zero {
                    data[column * rows + row] = T::zero();
                }
            }
        }

        Self::from_vec(rows, triangle, data)
    }
}

impl<T> From<FullTriangular<T>> for DMatrix<T>
where
    T: Scalar,
{
    fn from(matrix: FullTriangular<T>) -> Self {
        matrix.into_dmatrix()
    }
}

macro_rules! impl_packed_to_dmatrix {
    ($packed:ident) => {
        #[allow(private_bounds)]
        impl<T, S> $packed<T, S>
        where
            T: PackedFormatConversion + Scalar,
            S: PackedStorage<T>,
        {
            /// Converts traditional packed (`TP`) storage to an owned nalgebra matrix.
            ///
            /// LAPACK first expands the selected triangle to a full triangular (`TR`)
            /// buffer with structural zeros in the opposite triangle. The compatible
            /// column-major buffer is then moved into nalgebra. This allocates `n * n`
            /// entries; traditional packed storage cannot be exposed as a zero-copy
            /// `DMatrix` view.
            pub fn to_dmatrix(&self) -> Result<DMatrix<T>, PackedMatrixError> {
                Ok(self.to_full_triangular()?.into_dmatrix())
            }
        }
    };
}

impl_packed_to_dmatrix!(PackedLower);
impl_packed_to_dmatrix!(PackedUpper);

#[allow(private_bounds)]
impl<T> PackedLower<T>
where
    T: PackedFormatConversion + Scalar + Zero,
{
    /// Extracts the lower triangle of a square nalgebra matrix into owned packed storage.
    ///
    /// Values above the diagonal are intentionally discarded, not validated. LAPACK
    /// converts the resulting full triangular (`TR`) buffer to traditional packed
    /// (`TP`) storage. The layouts are incompatible, so this conversion allocates.
    pub fn from_lower_triangle(matrix: &DMatrix<T>) -> Result<Self, PackedMatrixError> {
        let full = FullTriangular::try_from_dmatrix(matrix, Triangle::Lower)?;
        Self::from_full_triangular(&full)
    }
}

#[allow(private_bounds)]
impl<T> PackedUpper<T>
where
    T: PackedFormatConversion + Scalar + Zero,
{
    /// Extracts the upper triangle of a square nalgebra matrix into owned packed storage.
    ///
    /// Values below the diagonal are intentionally discarded, not validated. LAPACK
    /// converts the resulting full triangular (`TR`) buffer to traditional packed
    /// (`TP`) storage. The layouts are incompatible, so this conversion allocates.
    pub fn from_upper_triangle(matrix: &DMatrix<T>) -> Result<Self, PackedMatrixError> {
        let full = FullTriangular::try_from_dmatrix(matrix, Triangle::Upper)?;
        Self::from_full_triangular(&full)
    }
}

trait StructuredScalar: Scalar + Copy + Zero {
    fn conjugate(self) -> Self;
    fn hermitian_diagonal(self) -> Self;
}

macro_rules! impl_real_structured_scalar {
    ($type:ty) => {
        impl StructuredScalar for $type {
            fn conjugate(self) -> Self {
                self
            }

            fn hermitian_diagonal(self) -> Self {
                self
            }
        }
    };
}

impl_real_structured_scalar!(f32);
impl_real_structured_scalar!(f64);

macro_rules! impl_complex_structured_scalar {
    ($type:ty) => {
        impl StructuredScalar for $type {
            fn conjugate(self) -> Self {
                self.conj()
            }

            fn hermitian_diagonal(self) -> Self {
                Self::new(self.re, 0.0)
            }
        }
    };
}

impl_complex_structured_scalar!(Complex32);
impl_complex_structured_scalar!(Complex64);

#[derive(Clone, Copy)]
enum Reconstruction {
    Symmetric,
    Hermitian,
}

fn expand_lower_packed<T: StructuredScalar>(
    dimension: usize,
    packed: &[T],
    reconstruction: Reconstruction,
) -> DMatrix<T> {
    let mut data = vec![T::zero(); dimension * dimension];
    let mut packed_index = 0;
    for column in 0..dimension {
        for row in column..dimension {
            let value = packed[packed_index];
            packed_index += 1;
            if row == column {
                data[column * dimension + row] = match reconstruction {
                    Reconstruction::Symmetric => value,
                    Reconstruction::Hermitian => value.hermitian_diagonal(),
                };
            } else {
                data[column * dimension + row] = value;
                data[row * dimension + column] = match reconstruction {
                    Reconstruction::Symmetric => value,
                    Reconstruction::Hermitian => value.conjugate(),
                };
            }
        }
    }
    DMatrix::from_vec(dimension, dimension, data)
}

fn extract_lower<T: StructuredScalar>(
    matrix: &DMatrix<T>,
    canonicalize_diagonal: bool,
) -> Result<Vec<T>, PackedMatrixError> {
    let (rows, columns) = matrix.shape();
    if rows != columns {
        return Err(PackedMatrixError::NonSquareMatrix { rows, columns });
    }

    let mut packed = Vec::with_capacity(rows * (rows + 1) / 2);
    for column in 0..columns {
        for row in column..rows {
            let value = matrix[(row, column)];
            packed.push(if canonicalize_diagonal && row == column {
                value.hermitian_diagonal()
            } else {
                value
            });
        }
    }
    Ok(packed)
}

#[allow(private_bounds)]
impl<T, S> PackedSymmetric<T, S>
where
    T: StructuredScalar,
    S: PackedStorage<T>,
{
    /// Expands lower packed storage into a complete logical symmetric matrix.
    ///
    /// Both triangles are allocated in the result. Complex values are mirrored
    /// without conjugation, so complex symmetric matrices remain distinct from
    /// Hermitian matrices. Packed storage cannot be exposed as a zero-copy view.
    pub fn to_dmatrix(&self) -> DMatrix<T> {
        expand_lower_packed(self.dimension(), self.as_slice(), Reconstruction::Symmetric)
    }
}

#[allow(private_bounds)]
impl<T> PackedSymmetric<T>
where
    T: StructuredScalar,
{
    /// Extracts the lower triangle of a square matrix without validating symmetry.
    ///
    /// The upper triangle is intentionally ignored and the result owns newly
    /// allocated traditional packed storage.
    pub fn from_lower_triangle(matrix: &DMatrix<T>) -> Result<Self, PackedMatrixError> {
        Self::from_vec(matrix.nrows(), extract_lower(matrix, false)?)
    }
}

#[allow(private_bounds)]
impl<T, S> PackedHermitian<T, S>
where
    T: StructuredScalar,
    S: PackedStorage<T>,
{
    /// Expands lower packed storage into a complete logical Hermitian matrix.
    ///
    /// Upper entries are conjugated and diagonal imaginary components are
    /// discarded according to LAPACK's real-diagonal convention. The result
    /// allocates full storage and is not a zero-copy view.
    pub fn to_dmatrix(&self) -> DMatrix<T> {
        expand_lower_packed(self.dimension(), self.as_slice(), Reconstruction::Hermitian)
    }
}

#[allow(private_bounds)]
impl<T> PackedHermitian<T>
where
    T: StructuredScalar,
{
    /// Extracts the lower triangle of a square matrix without validating Hermitian structure.
    ///
    /// The upper triangle is intentionally ignored. Diagonal imaginary components
    /// are discarded to retain LAPACK's real-diagonal convention.
    pub fn from_lower_triangle(matrix: &DMatrix<T>) -> Result<Self, PackedMatrixError> {
        Self::from_vec(matrix.nrows(), extract_lower(matrix, true)?)
    }
}

#[allow(private_bounds)]
impl<T, S> PackedSPD<T, S>
where
    T: StructuredScalar,
    S: PackedStorage<T>,
{
    /// Expands lower packed SPD/HPD storage into a complete logical matrix.
    ///
    /// Real values are reconstructed symmetrically; complex values use Hermitian
    /// conjugation and a real diagonal. The conversion allocates `n * n` entries
    /// and cannot provide a zero-copy view.
    pub fn to_dmatrix(&self) -> DMatrix<T> {
        expand_lower_packed(self.dimension(), self.as_slice(), Reconstruction::Hermitian)
    }
}

#[allow(private_bounds)]
impl<T> PackedSPD<T>
where
    T: StructuredScalar,
{
    /// Extracts lower storage without proving symmetric/Hermitian or positive-definite structure.
    ///
    /// The upper triangle is intentionally ignored. Complex diagonal imaginary
    /// components are discarded according to LAPACK's HPD convention. Numerical
    /// invalidity is possible but does not affect memory safety.
    pub fn from_lower_triangle_unchecked_structure(
        matrix: &DMatrix<T>,
    ) -> Result<Self, PackedMatrixError> {
        Self::from_vec(matrix.nrows(), extract_lower(matrix, true)?)
    }
}
