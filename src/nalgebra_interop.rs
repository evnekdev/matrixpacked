//! Optional conversions between crate-owned full triangular storage and nalgebra.

use crate::{FullTriangular, PackedMatrixError, Triangle};
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
