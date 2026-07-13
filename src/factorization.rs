//! Reusable packed factorizations preserving the caller's storage ownership.

use crate::{
    PackedMatrixError,
    backend::{HermitianPackedBackend, PositiveDefinitePackedBackend, SymmetricPackedBackend},
    storage::{PackedStorage, PackedStorageMut},
};
use num_traits::Zero;

pub(crate) fn checked_n(n: usize) -> Result<i32, PackedMatrixError> {
    i32::try_from(n).map_err(|_| PackedMatrixError::DimensionOverflow { n })
}
pub(crate) fn checked_workspace_len(
    n: usize,
    multiplier: usize,
) -> Result<usize, PackedMatrixError> {
    n.checked_mul(multiplier)
        .ok_or(PackedMatrixError::DimensionOverflow { n })
}
pub(crate) fn check_rhs<T>(n: usize, rhs: &[T]) -> Result<(), PackedMatrixError> {
    if rhs.len() == n {
        Ok(())
    } else {
        Err(PackedMatrixError::InvalidVectorLength {
            expected: n,
            actual: rhs.len(),
        })
    }
}
pub(crate) fn check_rhs_many<T>(n: usize, nrhs: usize, rhs: &[T]) -> Result<(), PackedMatrixError> {
    let expected = n
        .checked_mul(nrhs)
        .ok_or(PackedMatrixError::DimensionOverflow { n })?;
    if rhs.len() == expected {
        Ok(())
    } else {
        Err(PackedMatrixError::InvalidVectorLength {
            expected,
            actual: rhs.len(),
        })
    }
}
pub(crate) fn check_info(info: i32, message: &'static str) -> Result<(), PackedMatrixError> {
    if info < 0 {
        Err(PackedMatrixError::LapackIllegalArgument { argument: -info })
    } else if info > 0 {
        Err(PackedMatrixError::FactorizationFailure {
            index: info as usize,
            message,
        })
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct PackedCholesky<T, S = Vec<T>> {
    pub(crate) n: usize,
    pub(crate) data: S,
    pub(crate) uplo: u8,
    pub(crate) marker: std::marker::PhantomData<T>,
}
pub type PackedCholeskyViewMut<'a, T> = PackedCholesky<T, &'a mut [T]>;
impl<T, S> PackedCholesky<T, S>
where
    T: PositiveDefinitePackedBackend,
    S: PackedStorageMut<T>,
{
    pub(crate) fn factorize_storage(
        n: usize,
        mut data: S,
        uplo: u8,
    ) -> Result<Self, PackedMatrixError> {
        let mut info = 0;
        unsafe { T::pptrf(uplo, checked_n(n)?, data.as_mut_slice(), &mut info) };
        check_info(info, "matrix is not positive definite")?;
        Ok(Self {
            n,
            data,
            uplo,
            marker: std::marker::PhantomData,
        })
    }
    /// Overwrites the Cholesky factor with the packed inverse.
    ///
    /// After success this value no longer contains a Cholesky factor; prefer
    /// [`Self::into_inverse`] for owned factors when an explicit type transition is possible.
    pub fn inverse_in_place(&mut self) -> Result<(), PackedMatrixError> {
        let mut info = 0;
        unsafe {
            T::pptri(
                self.uplo,
                checked_n(self.n)?,
                self.data.as_mut_slice(),
                &mut info,
            )
        };
        check_info(info, "packed Cholesky inverse failed")
    }
}
impl<T, S> PackedCholesky<T, S>
where
    T: PositiveDefinitePackedBackend,
    S: PackedStorage<T>,
{
    pub fn dimension(&self) -> usize {
        self.n
    }
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }
    /// Estimates reciprocal one-norm condition from this factorization. `anorm` is the original matrix one-norm.
    pub fn rcond(&self, anorm: T::Real) -> Result<T::Real, PackedMatrixError> {
        let mut r = T::Real::zero();
        let mut work =
            vec![T::zero(); checked_workspace_len(self.n, if T::IS_COMPLEX { 2 } else { 3 })?];
        let mut rw = vec![T::Real::zero(); if T::IS_COMPLEX { self.n } else { 0 }];
        let mut iw = vec![0; if T::IS_COMPLEX { 0 } else { self.n }];
        let mut info = 0;
        unsafe {
            T::ppcon(
                self.uplo,
                checked_n(self.n)?,
                self.as_slice(),
                anorm,
                &mut r,
                &mut work,
                &mut rw,
                &mut iw,
                &mut info,
            )
        };
        check_info(info, "packed Cholesky condition estimate failed")?;
        Ok(r)
    }
    pub fn solve_vector_in_place(&self, rhs: &mut [T]) -> Result<(), PackedMatrixError> {
        self.solve_many_in_place(rhs, 1)
    }
    pub fn solve_many_in_place(&self, rhs: &mut [T], nrhs: usize) -> Result<(), PackedMatrixError> {
        check_rhs_many(self.n, nrhs, rhs)?;
        let n = checked_n(self.n)?;
        let mut info = 0;
        unsafe {
            T::pptrs(
                self.uplo,
                n,
                checked_n(nrhs)?,
                self.as_slice(),
                rhs,
                n,
                &mut info,
            )
        };
        check_info(info, "packed Cholesky solve failed")
    }
    pub fn solve_vector(&self, rhs: &[T]) -> Result<Vec<T>, PackedMatrixError> {
        let mut out = rhs.to_vec();
        self.solve_vector_in_place(&mut out)?;
        Ok(out)
    }
    /// Refines `x` in place for column-major `n × nrhs` right-hand sides and returns one forward and backward error estimate per right-hand side.
    pub fn refine_many_in_place<O: PackedStorage<T>>(
        &self,
        original: &crate::PackedSPD<T, O>,
        b: &[T],
        x: &mut [T],
        nrhs: usize,
    ) -> Result<crate::RefinementReport<T::Real>, PackedMatrixError> {
        if original.dimension() != self.n {
            return Err(PackedMatrixError::DimensionMismatch {
                left: original.dimension(),
                right: self.n,
            });
        }
        check_rhs_many(self.n, nrhs, b)?;
        check_rhs_many(self.n, nrhs, x)?;
        let n = checked_n(self.n)?;
        let nr = checked_n(nrhs)?;
        let ld = n.max(1);
        let mut ferr = vec![T::Real::zero(); nrhs];
        let mut berr = ferr.clone();
        let mut work =
            vec![T::zero(); checked_workspace_len(self.n, if T::IS_COMPLEX { 2 } else { 3 })?];
        let mut rw = vec![T::Real::zero(); if T::IS_COMPLEX { self.n } else { 0 }];
        let mut iw = vec![0; if T::IS_COMPLEX { 0 } else { self.n }];
        let mut info = 0;
        unsafe {
            T::pprfs(
                self.uplo,
                n,
                nr,
                original.as_slice(),
                self.as_slice(),
                b,
                ld,
                x,
                ld,
                &mut ferr,
                &mut berr,
                &mut work,
                &mut rw,
                &mut iw,
                &mut info,
            )
        }
        check_info(info, "packed Cholesky iterative refinement failed")?;
        Ok(crate::RefinementReport {
            forward_error: ferr,
            backward_error: berr,
        })
    }
    pub fn refine_vector_in_place<O: PackedStorage<T>>(
        &self,
        original: &crate::PackedSPD<T, O>,
        b: &[T],
        x: &mut [T],
    ) -> Result<crate::RefinementReport<T::Real>, PackedMatrixError> {
        self.refine_many_in_place(original, b, x, 1)
    }
}
impl<T> PackedCholesky<T, Vec<T>> {
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}
impl<T> PackedCholesky<T, Vec<T>>
where
    T: PositiveDefinitePackedBackend,
{
    /// Consumes the factorization and returns the inverse in packed structured storage.
    pub fn into_inverse(mut self) -> Result<crate::PackedSPD<T>, PackedMatrixError> {
        self.inverse_in_place()?;
        crate::PackedSPD::from_vec(self.n, self.data)
    }
}

#[derive(Clone, Debug)]
pub struct PackedSymmetricFactor<T, S = Vec<T>> {
    pub(crate) n: usize,
    pub(crate) data: S,
    pub(crate) pivots: Vec<i32>,
    pub(crate) uplo: u8,
    pub(crate) marker: std::marker::PhantomData<T>,
}
pub type PackedSymmetricFactorViewMut<'a, T> = PackedSymmetricFactor<T, &'a mut [T]>;
impl<T, S> PackedSymmetricFactor<T, S>
where
    T: SymmetricPackedBackend,
    S: PackedStorageMut<T>,
{
    pub(crate) fn factorize_storage(
        n: usize,
        mut data: S,
        uplo: u8,
    ) -> Result<Self, PackedMatrixError> {
        let mut pivots = vec![0; n];
        let mut info = 0;
        unsafe {
            T::sptrf(
                uplo,
                checked_n(n)?,
                data.as_mut_slice(),
                &mut pivots,
                &mut info,
            )
        };
        check_info(info, "symmetric packed matrix is singular")?;
        Ok(Self {
            n,
            data,
            pivots,
            uplo,
            marker: std::marker::PhantomData,
        })
    }
    /// Overwrites the factorization with the packed symmetric inverse.
    /// Prefer [`Self::into_inverse`] for owned factors.
    pub fn inverse_in_place(&mut self) -> Result<(), PackedMatrixError> {
        let mut work = vec![T::zero(); self.n];
        let mut info = 0;
        unsafe {
            T::sptri(
                self.uplo,
                checked_n(self.n)?,
                self.data.as_mut_slice(),
                &self.pivots,
                &mut work,
                &mut info,
            )
        };
        check_info(info, "symmetric packed inverse failed")
    }
}
impl<T, S> PackedSymmetricFactor<T, S>
where
    T: SymmetricPackedBackend,
    S: PackedStorage<T>,
{
    pub fn dimension(&self) -> usize {
        self.n
    }
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }
    pub fn pivots(&self) -> &[i32] {
        &self.pivots
    }
    /// Estimates reciprocal one-norm condition from this factorization. `anorm` is the original matrix one-norm.
    pub fn rcond(&self, anorm: T::Real) -> Result<T::Real, PackedMatrixError> {
        let mut r = T::Real::zero();
        let mut work = vec![T::zero(); checked_workspace_len(self.n, 2)?];
        let mut iw = vec![0; if T::IS_COMPLEX { 0 } else { self.n }];
        let mut info = 0;
        unsafe {
            T::spcon(
                self.uplo,
                checked_n(self.n)?,
                self.as_slice(),
                &self.pivots,
                anorm,
                &mut r,
                &mut work,
                &mut iw,
                &mut info,
            )
        };
        check_info(info, "symmetric packed condition estimate failed")?;
        Ok(r)
    }
    pub fn solve_vector_in_place(&self, rhs: &mut [T]) -> Result<(), PackedMatrixError> {
        self.solve_many_in_place(rhs, 1)
    }
    pub fn solve_many_in_place(&self, rhs: &mut [T], nrhs: usize) -> Result<(), PackedMatrixError> {
        check_rhs_many(self.n, nrhs, rhs)?;
        let n = checked_n(self.n)?;
        let mut info = 0;
        unsafe {
            T::sptrs(
                self.uplo,
                n,
                checked_n(nrhs)?,
                self.as_slice(),
                &self.pivots,
                rhs,
                n,
                &mut info,
            )
        };
        check_info(info, "symmetric packed solve failed")
    }
    pub fn solve_vector(&self, rhs: &[T]) -> Result<Vec<T>, PackedMatrixError> {
        let mut out = rhs.to_vec();
        self.solve_vector_in_place(&mut out)?;
        Ok(out)
    }
    /// Refines `x` in place for column-major `n × nrhs` right-hand sides and returns one error estimate of each kind per right-hand side.
    pub fn refine_many_in_place<O: PackedStorage<T>>(
        &self,
        original: &crate::PackedSymmetric<T, O>,
        b: &[T],
        x: &mut [T],
        nrhs: usize,
    ) -> Result<crate::RefinementReport<T::Real>, PackedMatrixError> {
        if original.dimension() != self.n {
            return Err(PackedMatrixError::DimensionMismatch {
                left: original.dimension(),
                right: self.n,
            });
        }
        check_rhs_many(self.n, nrhs, b)?;
        check_rhs_many(self.n, nrhs, x)?;
        let n = checked_n(self.n)?;
        let nr = checked_n(nrhs)?;
        let ld = n.max(1);
        let mut ferr = vec![T::Real::zero(); nrhs];
        let mut berr = ferr.clone();
        let mut work =
            vec![T::zero(); checked_workspace_len(self.n, if T::IS_COMPLEX { 2 } else { 3 })?];
        let mut rw = vec![T::Real::zero(); if T::IS_COMPLEX { self.n } else { 0 }];
        let mut iw = vec![0; if T::IS_COMPLEX { 0 } else { self.n }];
        let mut info = 0;
        unsafe {
            T::sprfs(
                self.uplo,
                n,
                nr,
                original.as_slice(),
                self.as_slice(),
                &self.pivots,
                b,
                ld,
                x,
                ld,
                &mut ferr,
                &mut berr,
                &mut work,
                &mut rw,
                &mut iw,
                &mut info,
            )
        }
        check_info(info, "symmetric packed iterative refinement failed")?;
        Ok(crate::RefinementReport {
            forward_error: ferr,
            backward_error: berr,
        })
    }
    pub fn refine_vector_in_place<O: PackedStorage<T>>(
        &self,
        original: &crate::PackedSymmetric<T, O>,
        b: &[T],
        x: &mut [T],
    ) -> Result<crate::RefinementReport<T::Real>, PackedMatrixError> {
        self.refine_many_in_place(original, b, x, 1)
    }
}
impl<T> PackedSymmetricFactor<T, Vec<T>> {
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}
impl<T> PackedSymmetricFactor<T, Vec<T>>
where
    T: SymmetricPackedBackend,
{
    /// Consumes the factorization and returns the symmetric packed inverse.
    pub fn into_inverse(mut self) -> Result<crate::PackedSymmetric<T>, PackedMatrixError> {
        self.inverse_in_place()?;
        crate::PackedSymmetric::from_vec(self.n, self.data)
    }
}

#[derive(Clone, Debug)]
pub struct PackedHermitianFactor<T, S = Vec<T>> {
    pub(crate) n: usize,
    pub(crate) data: S,
    pub(crate) pivots: Vec<i32>,
    pub(crate) uplo: u8,
    pub(crate) marker: std::marker::PhantomData<T>,
}
pub type PackedHermitianFactorViewMut<'a, T> = PackedHermitianFactor<T, &'a mut [T]>;
impl<T, S> PackedHermitianFactor<T, S>
where
    T: HermitianPackedBackend,
    S: PackedStorageMut<T>,
{
    pub(crate) fn factorize_storage(
        n: usize,
        mut data: S,
        uplo: u8,
    ) -> Result<Self, PackedMatrixError> {
        let mut pivots = vec![0; n];
        let mut info = 0;
        unsafe {
            T::hptrf(
                uplo,
                checked_n(n)?,
                data.as_mut_slice(),
                &mut pivots,
                &mut info,
            )
        };
        check_info(info, "Hermitian packed matrix is singular")?;
        Ok(Self {
            n,
            data,
            pivots,
            uplo,
            marker: std::marker::PhantomData,
        })
    }
    /// Overwrites the factorization with the packed Hermitian inverse.
    /// Prefer [`Self::into_inverse`] for owned factors.
    pub fn inverse_in_place(&mut self) -> Result<(), PackedMatrixError> {
        let mut work = vec![T::zero(); self.n];
        let mut info = 0;
        unsafe {
            T::hptri(
                self.uplo,
                checked_n(self.n)?,
                self.data.as_mut_slice(),
                &self.pivots,
                &mut work,
                &mut info,
            )
        };
        check_info(info, "Hermitian packed inverse failed")
    }
}
impl<T, S> PackedHermitianFactor<T, S>
where
    T: HermitianPackedBackend,
    S: PackedStorage<T>,
{
    pub fn dimension(&self) -> usize {
        self.n
    }
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }
    pub fn pivots(&self) -> &[i32] {
        &self.pivots
    }
    /// Estimates reciprocal one-norm condition from this factorization. `anorm` is the original matrix one-norm.
    pub fn rcond(&self, anorm: T::Real) -> Result<T::Real, PackedMatrixError> {
        let mut r = T::Real::zero();
        let mut work = vec![T::zero(); checked_workspace_len(self.n, 2)?];
        let mut info = 0;
        unsafe {
            T::hpcon(
                self.uplo,
                checked_n(self.n)?,
                self.as_slice(),
                &self.pivots,
                anorm,
                &mut r,
                &mut work,
                &mut info,
            )
        };
        check_info(info, "Hermitian packed condition estimate failed")?;
        Ok(r)
    }
    pub fn solve_vector_in_place(&self, rhs: &mut [T]) -> Result<(), PackedMatrixError> {
        self.solve_many_in_place(rhs, 1)
    }
    pub fn solve_many_in_place(&self, rhs: &mut [T], nrhs: usize) -> Result<(), PackedMatrixError> {
        check_rhs_many(self.n, nrhs, rhs)?;
        let n = checked_n(self.n)?;
        let mut info = 0;
        unsafe {
            T::hptrs(
                self.uplo,
                n,
                checked_n(nrhs)?,
                self.as_slice(),
                &self.pivots,
                rhs,
                n,
                &mut info,
            )
        };
        check_info(info, "Hermitian packed solve failed")
    }
    pub fn solve_vector(&self, rhs: &[T]) -> Result<Vec<T>, PackedMatrixError> {
        let mut out = rhs.to_vec();
        self.solve_vector_in_place(&mut out)?;
        Ok(out)
    }
    /// Refines `x` in place for column-major `n × nrhs` right-hand sides and returns one error estimate of each kind per right-hand side.
    pub fn refine_many_in_place<O: PackedStorage<T>>(
        &self,
        original: &crate::PackedHermitian<T, O>,
        b: &[T],
        x: &mut [T],
        nrhs: usize,
    ) -> Result<crate::RefinementReport<T::Real>, PackedMatrixError> {
        if original.dimension() != self.n {
            return Err(PackedMatrixError::DimensionMismatch {
                left: original.dimension(),
                right: self.n,
            });
        }
        check_rhs_many(self.n, nrhs, b)?;
        check_rhs_many(self.n, nrhs, x)?;
        let n = checked_n(self.n)?;
        let nr = checked_n(nrhs)?;
        let ld = n.max(1);
        let mut ferr = vec![T::Real::zero(); nrhs];
        let mut berr = ferr.clone();
        let mut work = vec![T::zero(); checked_workspace_len(self.n, 2)?];
        let mut rw = vec![T::Real::zero(); self.n];
        let mut info = 0;
        unsafe {
            T::hprfs(
                self.uplo,
                n,
                nr,
                original.as_slice(),
                self.as_slice(),
                &self.pivots,
                b,
                ld,
                x,
                ld,
                &mut ferr,
                &mut berr,
                &mut work,
                &mut rw,
                &mut info,
            )
        }
        check_info(info, "Hermitian packed iterative refinement failed")?;
        Ok(crate::RefinementReport {
            forward_error: ferr,
            backward_error: berr,
        })
    }
    pub fn refine_vector_in_place<O: PackedStorage<T>>(
        &self,
        original: &crate::PackedHermitian<T, O>,
        b: &[T],
        x: &mut [T],
    ) -> Result<crate::RefinementReport<T::Real>, PackedMatrixError> {
        self.refine_many_in_place(original, b, x, 1)
    }
}
impl<T> PackedHermitianFactor<T, Vec<T>> {
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}
impl<T> PackedHermitianFactor<T, Vec<T>>
where
    T: HermitianPackedBackend,
{
    /// Consumes the factorization and returns the Hermitian packed inverse.
    pub fn into_inverse(mut self) -> Result<crate::PackedHermitian<T>, PackedMatrixError> {
        self.inverse_in_place()?;
        crate::PackedHermitian::from_vec(self.n, self.data)
    }
}

#[cfg(test)]
mod refinement_tests {
    use crate::{PackedHermitian, PackedMatrixError, PackedSPD, PackedSymmetric};
    use num_complex::Complex64;

    #[test]
    fn pprfs_multiple_rhs_and_validation() {
        let a = PackedSPD::from_vec(2, vec![4.0f64, 1.0, 3.0]).unwrap();
        let factor = a.cholesky().unwrap();
        let b = [6.0, 7.0, 9.0, 5.0];
        let mut x = [0.9, 2.1, 2.1, 0.9];
        let report = factor.refine_many_in_place(&a, &b, &mut x, 2).unwrap();
        assert_eq!(report.forward_error.len(), 2);
        assert_eq!(report.backward_error.len(), 2);
        assert!(
            x.iter()
                .zip([1.0, 2.0, 2.0, 1.0])
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
        assert!(matches!(
            factor.refine_many_in_place(&a, &[1.0], &mut x, 2),
            Err(PackedMatrixError::InvalidVectorLength { .. })
        ));
        let mut short = [0.0];
        assert!(matches!(
            factor.refine_many_in_place(&a, &b, &mut short, 2),
            Err(PackedMatrixError::InvalidVectorLength { .. })
        ));
        let other = PackedSPD::from_vec(1, vec![1.0]).unwrap();
        assert!(matches!(
            factor.refine_many_in_place(&other, &b, &mut x, 2),
            Err(PackedMatrixError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn sprfs_one_rhs() {
        let a = PackedSymmetric::from_vec(2, vec![2.0f32, 1.0, -1.0]).unwrap();
        let f = a.factorize().unwrap();
        let b = [4.0, -1.0];
        let mut x = [0.8, 2.2];
        let r = f.refine_vector_in_place(&a, &b, &mut x).unwrap();
        assert_eq!(r.forward_error.len(), 1);
        assert!((x[0] - 1.).abs() < 1e-4 && (x[1] - 2.).abs() < 1e-4);
    }

    #[test]
    fn hprfs_complex_one_rhs() {
        let c = Complex64::new;
        let a = PackedHermitian::from_vec(2, vec![c(4., 0.), c(1., 1.), c(3., 0.)]).unwrap();
        let f = a.factorize().unwrap();
        let b = [c(5., -1.), c(4., 1.)];
        let mut x = [c(0.9, 0.), c(1.1, 0.)];
        let r = f.refine_vector_in_place(&a, &b, &mut x).unwrap();
        assert_eq!(r.backward_error.len(), 1);
        assert!((x[0] - c(1., 0.)).norm() < 1e-12);
    }

    #[test]
    fn zero_dimension_refinement() {
        let a = PackedSPD::<f64>::from_vec(0, vec![]).unwrap();
        let f = a.cholesky().unwrap();
        let mut x = [];
        let r = f.refine_many_in_place(&a, &[], &mut x, 0).unwrap();
        assert!(r.forward_error.is_empty());
    }
}
