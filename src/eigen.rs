//! Basic packed symmetric and Hermitian eigensolvers.

use num_traits::Zero;

use crate::{
    PackedHermitian, PackedMatrixError, PackedSymmetric,
    backend::{
        HermitianPackedDivideConquer, HermitianPackedEigen, SymmetricPackedDivideConquer,
        SymmetricPackedEigen,
    },
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

fn integer_workspace(value: i32, name: &'static str) -> Result<usize, PackedMatrixError> {
    usize::try_from(value)
        .ok()
        .filter(|&v| v > 0)
        .ok_or(PackedMatrixError::InvalidWorkspaceRecommendation { workspace: name })
}
fn workspace_i32(value: usize, name: &'static str) -> Result<i32, PackedMatrixError> {
    i32::try_from(value)
        .map_err(|_| PackedMatrixError::InvalidWorkspaceRecommendation { workspace: name })
}

fn symmetric_divide_conquer<T: SymmetricPackedDivideConquer>(
    n: usize,
    mut ap: Vec<T>,
    choice: Eigenvectors,
    uplo: u8,
) -> Result<EigenDecomposition<T, T>, PackedMatrixError> {
    let ni = checked_n(n)?;
    if n == 0 {
        return finish(
            n,
            Vec::new(),
            (choice == Eigenvectors::Compute).then(Vec::new),
            0,
        );
    }
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
    let mut work = vec![T::zero(); 1];
    let mut iwork = vec![0; 1];
    let mut info = 0;
    unsafe {
        T::spevd(
            if compute { b'V' } else { b'N' },
            uplo,
            ni,
            &mut ap,
            &mut w,
            &mut z,
            ni.max(1),
            &mut work,
            -1,
            &mut iwork,
            -1,
            &mut info,
        )
    };
    if info != 0 {
        return finish(n, w, None, info);
    }
    let lw = T::workspace_recommendation(work[0])
        .ok_or(PackedMatrixError::InvalidWorkspaceRecommendation { workspace: "WORK" })?;
    let liw = integer_workspace(iwork[0], "IWORK")?;
    work.resize(lw, T::zero());
    iwork.resize(liw, 0);
    unsafe {
        T::spevd(
            if compute { b'V' } else { b'N' },
            uplo,
            ni,
            &mut ap,
            &mut w,
            &mut z,
            ni.max(1),
            &mut work,
            workspace_i32(lw, "WORK")?,
            &mut iwork,
            workspace_i32(liw, "IWORK")?,
            &mut info,
        )
    };
    finish(n, w, compute.then_some(z), info)
}
fn hermitian_divide_conquer<T: HermitianPackedDivideConquer>(
    n: usize,
    mut ap: Vec<T>,
    choice: Eigenvectors,
    uplo: u8,
) -> Result<EigenDecomposition<T, T::Real>, PackedMatrixError> {
    let ni = checked_n(n)?;
    if n == 0 {
        return finish(
            n,
            Vec::new(),
            (choice == Eigenvectors::Compute).then(Vec::new),
            0,
        );
    }
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
    let mut work = vec![T::zero(); 1];
    let mut rwork = vec![T::Real::zero(); 1];
    let mut iwork = vec![0; 1];
    let mut info = 0;
    unsafe {
        T::hpevd(
            if compute { b'V' } else { b'N' },
            uplo,
            ni,
            &mut ap,
            &mut w,
            &mut z,
            ni.max(1),
            &mut work,
            -1,
            &mut rwork,
            -1,
            &mut iwork,
            -1,
            &mut info,
        )
    };
    if info != 0 {
        return finish(n, w, None, info);
    }
    let lw = T::work_recommendation(work[0])
        .ok_or(PackedMatrixError::InvalidWorkspaceRecommendation { workspace: "WORK" })?;
    let lrw = T::real_work_recommendation(rwork[0])
        .ok_or(PackedMatrixError::InvalidWorkspaceRecommendation { workspace: "RWORK" })?;
    let liw = integer_workspace(iwork[0], "IWORK")?;
    work.resize(lw, T::zero());
    rwork.resize(lrw, T::Real::zero());
    iwork.resize(liw, 0);
    unsafe {
        T::hpevd(
            if compute { b'V' } else { b'N' },
            uplo,
            ni,
            &mut ap,
            &mut w,
            &mut z,
            ni.max(1),
            &mut work,
            workspace_i32(lw, "WORK")?,
            &mut rwork,
            workspace_i32(lrw, "RWORK")?,
            &mut iwork,
            workspace_i32(liw, "IWORK")?,
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

impl<T: SymmetricPackedDivideConquer, S: PackedStorage<T>> PackedSymmetric<T, S> {
    /// Computes eigenvalues with divide-and-conquer `xSPEVD` and queried workspaces.
    pub fn eigenvalues_divide_conquer(&self) -> Result<Vec<T>, PackedMatrixError> {
        Ok(symmetric_divide_conquer(
            self.dimension(),
            self.as_slice().to_vec(),
            Eigenvectors::None,
            b'L',
        )?
        .eigenvalues)
    }
    /// Computes eigenpairs with divide-and-conquer `xSPEVD`.
    pub fn eigendecomposition_divide_conquer(
        &self,
    ) -> Result<EigenDecomposition<T, T>, PackedMatrixError> {
        symmetric_divide_conquer(
            self.dimension(),
            self.as_slice().to_vec(),
            Eigenvectors::Compute,
            b'L',
        )
    }
}
impl<T: SymmetricPackedDivideConquer> PackedSymmetric<T, Vec<T>> {
    pub fn into_eigendecomposition_divide_conquer(
        self,
    ) -> Result<EigenDecomposition<T, T>, PackedMatrixError> {
        let n = self.dimension();
        symmetric_divide_conquer(n, self.into_vec(), Eigenvectors::Compute, b'L')
    }
}
impl<T: HermitianPackedDivideConquer, S: PackedStorage<T>> PackedHermitian<T, S> {
    /// Computes eigenvalues with divide-and-conquer `xHPEVD` and queried workspaces.
    pub fn eigenvalues_divide_conquer(&self) -> Result<Vec<T::Real>, PackedMatrixError> {
        Ok(hermitian_divide_conquer(
            self.dimension(),
            self.as_slice().to_vec(),
            Eigenvectors::None,
            b'L',
        )?
        .eigenvalues)
    }
    /// Computes eigenpairs with divide-and-conquer `xHPEVD`.
    pub fn eigendecomposition_divide_conquer(
        &self,
    ) -> Result<EigenDecomposition<T, T::Real>, PackedMatrixError> {
        hermitian_divide_conquer(
            self.dimension(),
            self.as_slice().to_vec(),
            Eigenvectors::Compute,
            b'L',
        )
    }
}
impl<T: HermitianPackedDivideConquer> PackedHermitian<T, Vec<T>> {
    pub fn into_eigendecomposition_divide_conquer(
        self,
    ) -> Result<EigenDecomposition<T, T::Real>, PackedMatrixError> {
        let n = self.dimension();
        hermitian_divide_conquer(n, self.into_vec(), Eigenvectors::Compute, b'L')
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
    #[test]
    fn divide_conquer_real_agrees_with_basic() {
        let a = PackedSymmetric::from_vec(3, vec![2_f64, 1., 0., 2., 1., 2.]).unwrap();
        let basic = a.eigenvalues().unwrap();
        let dc = a.eigenvalues_divide_conquer().unwrap();
        for (a, b) in basic.into_iter().zip(dc) {
            close(a, b)
        }
        assert_eq!(
            a.eigendecomposition_divide_conquer()
                .unwrap()
                .eigenvectors
                .unwrap()
                .len(),
            9
        );
    }
    #[test]
    fn divide_conquer_complex_agrees_with_basic() {
        let c = Complex64::new;
        let a = PackedHermitian::from_vec(2, vec![c(2., 0.), c(0., 1.), c(2., 0.)]).unwrap();
        let basic = a.eigenvalues().unwrap();
        let dc = a.eigenvalues_divide_conquer().unwrap();
        for (a, b) in basic.into_iter().zip(dc) {
            close(a, b)
        }
        assert_eq!(
            a.eigendecomposition_divide_conquer()
                .unwrap()
                .eigenvectors
                .unwrap()
                .len(),
            4
        );
    }
    #[test]
    fn divide_conquer_f32_repeated_empty_and_upper() {
        let a = PackedSymmetric::from_vec(2, vec![2_f32, 0., 2.]).unwrap();
        assert_eq!(a.eigenvalues_divide_conquer().unwrap(), vec![2., 2.]);
        let empty = PackedHermitian::<Complex32>::from_vec(0, vec![]).unwrap();
        assert!(empty.eigenvalues_divide_conquer().unwrap().is_empty());
        let e =
            symmetric_divide_conquer(2, vec![2_f64, 1., 2.], Eigenvectors::Compute, b'U').unwrap();
        close(e.eigenvalues[0], 1.);
    }
    #[test]
    fn rejects_invalid_workspace_recommendations() {
        assert!(
            <f64 as SymmetricPackedDivideConquer>::workspace_recommendation(f64::NAN).is_none()
        );
        assert!(<f64 as SymmetricPackedDivideConquer>::workspace_recommendation(0.).is_none());
        assert!(integer_workspace(-1, "IWORK").is_err());
    }
}
