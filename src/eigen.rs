//! Basic packed symmetric and Hermitian eigensolvers.

use num_traits::Zero;

use crate::{
    PackedHermitian, PackedMatrixError, PackedSymmetric,
    backend::{HermitianPackedEigen, SymmetricPackedEigen},
    factorization::checked_n,
    storage::PackedStorage,
};

/// Selects whether LAPACK computes eigenvectors as well as eigenvalues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Eigenvectors {
    None,
    Compute,
}

/// Eigenvalues and optional eigenvectors of an `n x n` packed matrix.
///
/// Eigenvalues are in ascending order. Eigenvectors are column-major; vector
/// `j` occupies `eigenvectors[j*n .. (j+1)*n]`.
#[derive(Clone, Debug, PartialEq)]
pub struct EigenDecomposition<T, R> {
    pub eigenvalues: Vec<R>,
    pub eigenvectors: Option<Vec<T>>,
    pub dimension: usize,
}

fn finish<T, R>(
    n: usize,
    values: Vec<R>,
    vectors: Option<Vec<T>>,
    info: i32,
) -> Result<EigenDecomposition<T, R>, PackedMatrixError> {
    if info < 0 {
        return Err(PackedMatrixError::LapackIllegalArgument { argument: -info });
    }
    if info > 0 {
        return Err(PackedMatrixError::EigenvalueConvergenceFailure {
            unconverged: info as usize,
        });
    }
    Ok(EigenDecomposition {
        eigenvalues: values,
        eigenvectors: vectors,
        dimension: n,
    })
}

fn symmetric<T: SymmetricPackedEigen>(
    n: usize,
    mut ap: Vec<T>,
    choice: Eigenvectors,
    uplo: u8,
) -> Result<EigenDecomposition<T, T>, PackedMatrixError> {
    let ni = checked_n(n)?;
    let compute = choice == Eigenvectors::Compute;
    let mut w = vec![T::zero(); n];
    let mut z = vec![
        T::zero();
        if compute {
            n.checked_mul(n)
                .ok_or(PackedMatrixError::DimensionOverflow { n })?
        } else {
            1
        }
    ];
    let mut work = vec![T::zero(); n.saturating_mul(3).saturating_sub(1).max(1)];
    let mut info = 0;
    unsafe {
        T::spev(
            if compute { b'V' } else { b'N' },
            uplo,
            ni,
            &mut ap,
            &mut w,
            &mut z,
            ni.max(1),
            &mut work,
            &mut info,
        )
    };
    finish(n, w, compute.then_some(z), info)
}

fn hermitian<T: HermitianPackedEigen>(
    n: usize,
    mut ap: Vec<T>,
    choice: Eigenvectors,
    uplo: u8,
) -> Result<EigenDecomposition<T, T::Real>, PackedMatrixError> {
    let ni = checked_n(n)?;
    let compute = choice == Eigenvectors::Compute;
    let mut w = vec![T::Real::zero(); n];
    let mut z = vec![
        T::zero();
        if compute {
            n.checked_mul(n)
                .ok_or(PackedMatrixError::DimensionOverflow { n })?
        } else {
            1
        }
    ];
    let mut work = vec![T::zero(); n.saturating_mul(2).saturating_sub(1).max(1)];
    let mut rwork = vec![T::Real::zero(); n.saturating_mul(3).saturating_sub(2).max(1)];
    let mut info = 0;
    unsafe {
        T::hpev(
            if compute { b'V' } else { b'N' },
            uplo,
            ni,
            &mut ap,
            &mut w,
            &mut z,
            ni.max(1),
            &mut work,
            &mut rwork,
            &mut info,
        )
    };
    finish(n, w, compute.then_some(z), info)
}

impl<T: SymmetricPackedEigen, S: PackedStorage<T>> PackedSymmetric<T, S> {
    /// Computes eigenvalues only, cloning only packed storage because LAPACK overwrites it.
    pub fn eigenvalues(&self) -> Result<Vec<T>, PackedMatrixError> {
        Ok(symmetric(
            self.dimension(),
            self.as_slice().to_vec(),
            Eigenvectors::None,
            b'L',
        )?
        .eigenvalues)
    }
    /// Computes all eigenvalues and orthonormal eigenvectors using `xSPEV`.
    pub fn eigendecomposition(&self) -> Result<EigenDecomposition<T, T>, PackedMatrixError> {
        symmetric(
            self.dimension(),
            self.as_slice().to_vec(),
            Eigenvectors::Compute,
            b'L',
        )
    }
    /// Computes the requested basic eigensolver output using `xSPEV`.
    pub fn eigen(
        &self,
        choice: Eigenvectors,
    ) -> Result<EigenDecomposition<T, T>, PackedMatrixError> {
        symmetric(self.dimension(), self.as_slice().to_vec(), choice, b'L')
    }
}
impl<T: SymmetricPackedEigen> PackedSymmetric<T, Vec<T>> {
    /// Consumes and reuses the packed allocation overwritten by `xSPEV`.
    pub fn into_eigen(
        self,
        choice: Eigenvectors,
    ) -> Result<EigenDecomposition<T, T>, PackedMatrixError> {
        let n = self.dimension();
        symmetric(n, self.into_vec(), choice, b'L')
    }
    pub fn into_eigendecomposition(self) -> Result<EigenDecomposition<T, T>, PackedMatrixError> {
        self.into_eigen(Eigenvectors::Compute)
    }
}
impl<T: HermitianPackedEigen, S: PackedStorage<T>> PackedHermitian<T, S> {
    /// Computes eigenvalues only, cloning only packed storage because LAPACK overwrites it.
    pub fn eigenvalues(&self) -> Result<Vec<T::Real>, PackedMatrixError> {
        Ok(hermitian(
            self.dimension(),
            self.as_slice().to_vec(),
            Eigenvectors::None,
            b'L',
        )?
        .eigenvalues)
    }
    /// Computes all eigenvalues and unitary eigenvectors using `xHPEV`.
    pub fn eigendecomposition(&self) -> Result<EigenDecomposition<T, T::Real>, PackedMatrixError> {
        hermitian(
            self.dimension(),
            self.as_slice().to_vec(),
            Eigenvectors::Compute,
            b'L',
        )
    }
    pub fn eigen(
        &self,
        choice: Eigenvectors,
    ) -> Result<EigenDecomposition<T, T::Real>, PackedMatrixError> {
        hermitian(self.dimension(), self.as_slice().to_vec(), choice, b'L')
    }
}
impl<T: HermitianPackedEigen> PackedHermitian<T, Vec<T>> {
    /// Consumes and reuses the packed allocation overwritten by `xHPEV`.
    pub fn into_eigen(
        self,
        choice: Eigenvectors,
    ) -> Result<EigenDecomposition<T, T::Real>, PackedMatrixError> {
        let n = self.dimension();
        hermitian(n, self.into_vec(), choice, b'L')
    }
    pub fn into_eigendecomposition(
        self,
    ) -> Result<EigenDecomposition<T, T::Real>, PackedMatrixError> {
        self.into_eigen(Eigenvectors::Compute)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}")
    }
    #[test]
    fn real_values_vectors_and_repeated() {
        let a = PackedSymmetric::from_vec(3, vec![2., 0., 0., 2., 0., 5.]).unwrap();
        assert_eq!(a.eigenvalues().unwrap(), vec![2., 2., 5.]);
        let e = a.eigendecomposition().unwrap();
        assert_eq!(e.eigenvectors.as_ref().unwrap().len(), 9);
    }
    #[test]
    fn real_f32_and_one() {
        let a = PackedSymmetric::from_vec(1, vec![3_f32]).unwrap();
        assert_eq!(a.into_eigendecomposition().unwrap().eigenvalues, vec![3.]);
    }
    #[test]
    fn complex_values_vectors() {
        let c = |r, i| Complex64::new(r, i);
        let a = PackedHermitian::from_vec(2, vec![c(2., 0.), c(0., 1.), c(2., 0.)]).unwrap();
        let e = a.eigendecomposition().unwrap();
        close(e.eigenvalues[0], 1.);
        close(e.eigenvalues[1], 3.);
        assert_eq!(e.eigenvectors.unwrap().len(), 4);
    }
    #[test]
    fn complex_f32_and_empty() {
        let a = PackedHermitian::<Complex32>::from_vec(0, vec![]).unwrap();
        assert!(a.eigenvalues().unwrap().is_empty());
    }
    #[test]
    fn upper_and_lower_lapack_storage() {
        let lower = vec![2., 1., 2.];
        let upper = vec![2., 1., 2.];
        for (ap, u) in [(lower, b'L'), (upper, b'U')] {
            let e = symmetric(2, ap, Eigenvectors::Compute, u).unwrap();
            close(e.eigenvalues[0], 1.);
            close(e.eigenvalues[1], 3.);
        }
    }
}
