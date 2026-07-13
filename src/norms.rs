//! LAPACK-backed norms for packed symmetric, positive-definite, and Hermitian matrices.

use num_complex::{Complex32, Complex64};
use num_traits::Zero;

use crate::{
    MatrixNorm, PackedHermitian, PackedMatrixError, PackedSPD, PackedSymmetric,
    storage::PackedStorage,
};

/// Internal dispatch for `xLANSP`, the real/complex symmetric packed norm family.
pub(crate) trait SymmetricPackedNormBackend: crate::LapackScalar {
    unsafe fn lansp(norm: u8, uplo: u8, n: i32, ap: &[Self], work: &mut [Self::Real])
    -> Self::Real;
}

impl SymmetricPackedNormBackend for f32 {
    unsafe fn lansp(
        norm: u8,
        uplo: u8,
        n: i32,
        ap: &[Self],
        work: &mut [Self::Real],
    ) -> Self::Real {
        unsafe { lapack::slansp(norm, uplo, n, ap, work) }
    }
}

impl SymmetricPackedNormBackend for f64 {
    unsafe fn lansp(
        norm: u8,
        uplo: u8,
        n: i32,
        ap: &[Self],
        work: &mut [Self::Real],
    ) -> Self::Real {
        unsafe { lapack::dlansp(norm, uplo, n, ap, work) }
    }
}

impl SymmetricPackedNormBackend for Complex32 {
    unsafe fn lansp(
        norm: u8,
        uplo: u8,
        n: i32,
        ap: &[Self],
        work: &mut [Self::Real],
    ) -> Self::Real {
        unsafe { lapack::clansp(norm, uplo, n, ap, work) }
    }
}

impl SymmetricPackedNormBackend for Complex64 {
    unsafe fn lansp(
        norm: u8,
        uplo: u8,
        n: i32,
        ap: &[Self],
        work: &mut [Self::Real],
    ) -> Self::Real {
        unsafe { lapack::zlansp(norm, uplo, n, ap, work) }
    }
}

/// Internal dispatch for `xLANHP`, the complex Hermitian packed norm family.
pub(crate) trait HermitianPackedNormBackend: crate::LapackScalar {
    unsafe fn lanhp(norm: u8, uplo: u8, n: i32, ap: &[Self], work: &mut [Self::Real])
    -> Self::Real;
}

impl HermitianPackedNormBackend for Complex32 {
    unsafe fn lanhp(
        norm: u8,
        uplo: u8,
        n: i32,
        ap: &[Self],
        work: &mut [Self::Real],
    ) -> Self::Real {
        unsafe { lapack::clanhp(norm, uplo, n, ap, work) }
    }
}

impl HermitianPackedNormBackend for Complex64 {
    unsafe fn lanhp(
        norm: u8,
        uplo: u8,
        n: i32,
        ap: &[Self],
        work: &mut [Self::Real],
    ) -> Self::Real {
        unsafe { lapack::zlanhp(norm, uplo, n, ap, work) }
    }
}

/// Internal dispatch selecting `xLANSP` for real SPD matrices and `xLANHP`
/// for complex Hermitian positive-definite matrices.
pub(crate) trait PositiveDefinitePackedNormBackend: crate::LapackScalar {
    unsafe fn lanpp(norm: u8, uplo: u8, n: i32, ap: &[Self], work: &mut [Self::Real])
    -> Self::Real;
}

impl PositiveDefinitePackedNormBackend for f32 {
    unsafe fn lanpp(
        norm: u8,
        uplo: u8,
        n: i32,
        ap: &[Self],
        work: &mut [Self::Real],
    ) -> Self::Real {
        unsafe { lapack::slansp(norm, uplo, n, ap, work) }
    }
}

impl PositiveDefinitePackedNormBackend for f64 {
    unsafe fn lanpp(
        norm: u8,
        uplo: u8,
        n: i32,
        ap: &[Self],
        work: &mut [Self::Real],
    ) -> Self::Real {
        unsafe { lapack::dlansp(norm, uplo, n, ap, work) }
    }
}

impl PositiveDefinitePackedNormBackend for Complex32 {
    unsafe fn lanpp(
        norm: u8,
        uplo: u8,
        n: i32,
        ap: &[Self],
        work: &mut [Self::Real],
    ) -> Self::Real {
        unsafe { lapack::clanhp(norm, uplo, n, ap, work) }
    }
}

impl PositiveDefinitePackedNormBackend for Complex64 {
    unsafe fn lanpp(
        norm: u8,
        uplo: u8,
        n: i32,
        ap: &[Self],
        work: &mut [Self::Real],
    ) -> Self::Real {
        unsafe { lapack::zlanhp(norm, uplo, n, ap, work) }
    }
}

fn checked_dimension(n: usize) -> Result<i32, PackedMatrixError> {
    crate::factorization::checked_n(n)
}

impl<T, S> PackedSymmetric<T, S>
where
    T: SymmetricPackedNormBackend,
    S: PackedStorage<T>,
{
    /// Computes a norm of the logical symmetric matrix using LAPACK `xLANSP`.
    pub fn matrix_norm(&self, norm: MatrixNorm) -> Result<T::Real, PackedMatrixError> {
        let mut work = vec![T::Real::zero(); self.dimension()];
        Ok(unsafe {
            T::lansp(
                norm.as_lapack(),
                b'L',
                checked_dimension(self.dimension())?,
                self.as_slice(),
                &mut work,
            )
        })
    }
}

impl<T, S> PackedHermitian<T, S>
where
    T: HermitianPackedNormBackend,
    S: PackedStorage<T>,
{
    /// Computes a norm of the logical Hermitian matrix using LAPACK `xLANHP`.
    pub fn matrix_norm(&self, norm: MatrixNorm) -> Result<T::Real, PackedMatrixError> {
        let mut work = vec![T::Real::zero(); self.dimension()];
        Ok(unsafe {
            T::lanhp(
                norm.as_lapack(),
                b'L',
                checked_dimension(self.dimension())?,
                self.as_slice(),
                &mut work,
            )
        })
    }
}

impl<T, S> PackedSPD<T, S>
where
    T: PositiveDefinitePackedNormBackend,
    S: PackedStorage<T>,
{
    /// Computes a norm of the logical SPD/HPD matrix using `xLANSP` or `xLANHP`.
    pub fn matrix_norm(&self, norm: MatrixNorm) -> Result<T::Real, PackedMatrixError> {
        let mut work = vec![T::Real::zero(); self.dimension()];
        Ok(unsafe {
            T::lanpp(
                norm.as_lapack(),
                b'L',
                checked_dimension(self.dimension())?,
                self.as_slice(),
                &mut work,
            )
        })
    }
}
