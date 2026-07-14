//! Optional conversions between crate-owned full triangular storage and nalgebra.

use crate::{
    FullTriangular, PackedLower, PackedMatrixError, PackedUpper, Triangle,
    backend::PackedFormatConversion, storage::PackedStorage,
};
use nalgebra::{DMatrix, Scalar};
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
