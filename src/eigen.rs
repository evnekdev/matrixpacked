//! Basic packed symmetric and Hermitian eigensolvers.

use num_traits::Zero;

use crate::{
    PackedHermitian, PackedMatrixError, PackedSPD, PackedSymmetric,
    backend::{
        GeneralizedPackedEigen, HermitianPackedDivideConquer, HermitianPackedEigen,
        HermitianPackedSelected, SymmetricPackedDivideConquer, SymmetricPackedEigen,
        SymmetricPackedSelected,
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

/// Selects eigenvalues by all values, a zero-based inclusive index range, or
/// LAPACK's half-open value interval `(lower, upper]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EigenRange<R> {
    All,
    Index { first: usize, last: usize },
    Value { lower: R, upper: R },
}

/// Selected eigenvalues and optional column-major eigenvectors.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectedEigenDecomposition<T, R> {
    pub eigenvalues: Vec<R>,
    pub eigenvectors: Option<Vec<T>>,
    pub dimension: usize,
    pub count: usize,
}

/// Mathematical form of a generalized symmetric/Hermitian definite eigenproblem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeneralizedEigenproblem {
    AxEqualsLambdaBx,
    ABxEqualsLambdaX,
    BAxEqualsLambdaX,
}
impl GeneralizedEigenproblem {
    fn itype(self) -> i32 {
        match self {
            Self::AxEqualsLambdaBx => 1,
            Self::ABxEqualsLambdaX => 2,
            Self::BAxEqualsLambdaX => 3,
        }
    }
}

fn selection<R: Copy + Zero>(
    n: usize,
    range: EigenRange<R>,
    finite: impl Fn(R) -> bool,
    less: impl Fn(R, R) -> bool,
) -> Result<(u8, R, R, i32, i32), PackedMatrixError> {
    match range {
        EigenRange::All => Ok((b'A', R::zero(), R::zero(), 0, 0)),
        EigenRange::Index { first, last } => {
            if first > last {
                return Err(PackedMatrixError::InvalidEigenRange {
                    reason: "first must not exceed last",
                });
            }
            if last >= n {
                return Err(PackedMatrixError::InvalidEigenRange {
                    reason: "index is outside the matrix",
                });
            }
            let il = i32::try_from(
                first
                    .checked_add(1)
                    .ok_or(PackedMatrixError::DimensionOverflow { n })?,
            )
            .map_err(|_| PackedMatrixError::DimensionOverflow { n })?;
            let iu = i32::try_from(
                last.checked_add(1)
                    .ok_or(PackedMatrixError::DimensionOverflow { n })?,
            )
            .map_err(|_| PackedMatrixError::DimensionOverflow { n })?;
            Ok((b'I', R::zero(), R::zero(), il, iu))
        }
        EigenRange::Value { lower, upper } => {
            if !finite(lower) || !finite(upper) {
                return Err(PackedMatrixError::InvalidEigenRange {
                    reason: "value bounds must be finite",
                });
            }
            if !less(lower, upper) {
                return Err(PackedMatrixError::InvalidEigenRange {
                    reason: "lower must be less than upper",
                });
            }
            Ok((b'V', lower, upper, 0, 0))
        }
    }
}

fn selected_finish<T, R>(
    n: usize,
    m: i32,
    mut w: Vec<R>,
    mut z: Option<Vec<T>>,
    ifail: &[i32],
    info: i32,
) -> Result<SelectedEigenDecomposition<T, R>, PackedMatrixError> {
    if info < 0 {
        return Err(PackedMatrixError::LapackIllegalArgument { argument: -info });
    }
    if info > 0 {
        let failed = ifail
            .iter()
            .take(info as usize)
            .filter_map(|&i| usize::try_from(i).ok().and_then(|v| v.checked_sub(1)))
            .collect();
        return Err(PackedMatrixError::EigenvectorConvergenceFailure { failed });
    }
    let count = usize::try_from(m)
        .map_err(|_| PackedMatrixError::EigenvalueConvergenceFailure { unconverged: 1 })?;
    w.truncate(count);
    if let Some(v) = z.as_mut() {
        v.truncate(
            count
                .checked_mul(n)
                .ok_or(PackedMatrixError::DimensionOverflow { n })?,
        )
    }
    Ok(SelectedEigenDecomposition {
        eigenvalues: w,
        eigenvectors: z,
        dimension: n,
        count,
    })
}

fn symmetric_selected<T: SymmetricPackedSelected>(
    n: usize,
    mut ap: Vec<T>,
    range: EigenRange<T>,
    choice: Eigenvectors,
    abstol: T,
    uplo: u8,
) -> Result<SelectedEigenDecomposition<T, T>, PackedMatrixError> {
    if n == 0 {
        return match range {
            EigenRange::All | EigenRange::Value { .. } => Ok(SelectedEigenDecomposition {
                eigenvalues: vec![],
                eigenvectors: (choice == Eigenvectors::Compute).then(Vec::new),
                dimension: 0,
                count: 0,
            }),
            EigenRange::Index { .. } => Err(PackedMatrixError::InvalidEigenRange {
                reason: "index range is invalid for an empty matrix",
            }),
        };
    }
    let ni = checked_n(n)?;
    if !T::finite(abstol) || T::less(abstol, T::zero()) {
        return Err(PackedMatrixError::InvalidEigenRange {
            reason: "absolute tolerance must be finite and nonnegative",
        });
    }
    let (r, vl, vu, il, iu) = selection(n, range, T::finite, T::less)?;
    let compute = choice == Eigenvectors::Compute;
    let mut m = 0;
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
    let mut work = vec![
        T::zero();
        n.checked_mul(8)
            .ok_or(PackedMatrixError::DimensionOverflow { n })?
    ];
    let mut iw = vec![
        0;
        n.checked_mul(5)
            .ok_or(PackedMatrixError::DimensionOverflow { n })?
    ];
    let mut fail = vec![0; n];
    let mut info = 0;
    unsafe {
        T::spevx(
            if compute { b'V' } else { b'N' },
            r,
            uplo,
            ni,
            &mut ap,
            vl,
            vu,
            il,
            iu,
            abstol,
            &mut m,
            &mut w,
            &mut z,
            ni.max(1),
            &mut work,
            &mut iw,
            &mut fail,
            &mut info,
        )
    }
    selected_finish(n, m, w, compute.then_some(z), &fail, info)
}

fn hermitian_selected<T: HermitianPackedSelected>(
    n: usize,
    mut ap: Vec<T>,
    range: EigenRange<T::Real>,
    choice: Eigenvectors,
    abstol: T::Real,
    uplo: u8,
) -> Result<SelectedEigenDecomposition<T, T::Real>, PackedMatrixError> {
    if n == 0 {
        return match range {
            EigenRange::All | EigenRange::Value { .. } => Ok(SelectedEigenDecomposition {
                eigenvalues: vec![],
                eigenvectors: (choice == Eigenvectors::Compute).then(Vec::new),
                dimension: 0,
                count: 0,
            }),
            EigenRange::Index { .. } => Err(PackedMatrixError::InvalidEigenRange {
                reason: "index range is invalid for an empty matrix",
            }),
        };
    }
    let ni = checked_n(n)?;
    if !T::finite(abstol) || T::less(abstol, T::Real::zero()) {
        return Err(PackedMatrixError::InvalidEigenRange {
            reason: "absolute tolerance must be finite and nonnegative",
        });
    }
    let (r, vl, vu, il, iu) = selection(n, range, T::finite, T::less)?;
    let compute = choice == Eigenvectors::Compute;
    let mut m = 0;
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
    let mut work = vec![
        T::zero();
        n.checked_mul(2)
            .ok_or(PackedMatrixError::DimensionOverflow { n })?
    ];
    let mut rw = vec![
        T::Real::zero();
        n.checked_mul(7)
            .ok_or(PackedMatrixError::DimensionOverflow { n })?
    ];
    let mut iw = vec![
        0;
        n.checked_mul(5)
            .ok_or(PackedMatrixError::DimensionOverflow { n })?
    ];
    let mut fail = vec![0; n];
    let mut info = 0;
    unsafe {
        T::hpevx(
            if compute { b'V' } else { b'N' },
            r,
            uplo,
            ni,
            &mut ap,
            vl,
            vu,
            il,
            iu,
            abstol,
            &mut m,
            &mut w,
            &mut z,
            ni.max(1),
            &mut work,
            &mut rw,
            &mut iw,
            &mut fail,
            &mut info,
        )
    }
    selected_finish(n, m, w, compute.then_some(z), &fail, info)
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

impl<T: SymmetricPackedSelected, S: PackedStorage<T>> PackedSymmetric<T, S> {
    pub fn selected_eigen(
        &self,
        range: EigenRange<T>,
        choice: Eigenvectors,
    ) -> Result<SelectedEigenDecomposition<T, T>, PackedMatrixError> {
        self.selected_eigen_with_abstol(range, choice, T::zero())
    }
    pub fn selected_eigen_with_abstol(
        &self,
        range: EigenRange<T>,
        choice: Eigenvectors,
        abstol: T,
    ) -> Result<SelectedEigenDecomposition<T, T>, PackedMatrixError> {
        symmetric_selected(
            self.dimension(),
            self.as_slice().to_vec(),
            range,
            choice,
            abstol,
            b'L',
        )
    }
    pub fn selected_eigenvalues(&self, range: EigenRange<T>) -> Result<Vec<T>, PackedMatrixError> {
        Ok(self.selected_eigen(range, Eigenvectors::None)?.eigenvalues)
    }
    pub fn selected_eigendecomposition(
        &self,
        range: EigenRange<T>,
    ) -> Result<SelectedEigenDecomposition<T, T>, PackedMatrixError> {
        self.selected_eigen(range, Eigenvectors::Compute)
    }
}
impl<T: HermitianPackedSelected, S: PackedStorage<T>> PackedHermitian<T, S> {
    pub fn selected_eigen(
        &self,
        range: EigenRange<T::Real>,
        choice: Eigenvectors,
    ) -> Result<SelectedEigenDecomposition<T, T::Real>, PackedMatrixError> {
        self.selected_eigen_with_abstol(range, choice, T::Real::zero())
    }
    pub fn selected_eigen_with_abstol(
        &self,
        range: EigenRange<T::Real>,
        choice: Eigenvectors,
        abstol: T::Real,
    ) -> Result<SelectedEigenDecomposition<T, T::Real>, PackedMatrixError> {
        hermitian_selected(
            self.dimension(),
            self.as_slice().to_vec(),
            range,
            choice,
            abstol,
            b'L',
        )
    }
    pub fn selected_eigenvalues(
        &self,
        range: EigenRange<T::Real>,
    ) -> Result<Vec<T::Real>, PackedMatrixError> {
        Ok(self.selected_eigen(range, Eigenvectors::None)?.eigenvalues)
    }
    pub fn selected_eigendecomposition(
        &self,
        range: EigenRange<T::Real>,
    ) -> Result<SelectedEigenDecomposition<T, T::Real>, PackedMatrixError> {
        self.selected_eigen(range, Eigenvectors::Compute)
    }
}

fn generalized_info(n: usize, info: i32) -> Result<(), PackedMatrixError> {
    if info < 0 {
        Err(PackedMatrixError::LapackIllegalArgument { argument: -info })
    } else if info > n as i32 {
        Err(PackedMatrixError::PositiveDefinitenessFailure {
            index: (info - n as i32) as usize,
        })
    } else if info > 0 {
        Err(PackedMatrixError::EigenvalueConvergenceFailure {
            unconverged: info as usize,
        })
    } else {
        Ok(())
    }
}
fn generalized_basic<T: GeneralizedPackedEigen>(
    n: usize,
    mut a: Vec<T>,
    mut b: Vec<T>,
    p: GeneralizedEigenproblem,
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
    let mut work = vec![
        T::zero();
        if T::COMPLEX {
            n.saturating_mul(2).saturating_sub(1).max(1)
        } else {
            n.saturating_mul(3).saturating_sub(1).max(1)
        }
    ];
    let mut rw = vec![
        T::Real::zero();
        if T::COMPLEX {
            n.saturating_mul(3).saturating_sub(2).max(1)
        } else {
            1
        }
    ];
    let mut info = 0;
    unsafe {
        T::pgv(
            &[p.itype()],
            if compute { b'V' } else { b'N' },
            uplo,
            ni,
            &mut a,
            &mut b,
            &mut w,
            &mut z,
            ni.max(1),
            &mut work,
            &mut rw,
            &mut info,
        )
    };
    generalized_info(n, info)?;
    Ok(EigenDecomposition {
        eigenvalues: w,
        eigenvectors: compute.then_some(z),
        dimension: n,
    })
}
fn generalized_dc<T: GeneralizedPackedEigen>(
    n: usize,
    mut a: Vec<T>,
    mut b: Vec<T>,
    p: GeneralizedEigenproblem,
    choice: Eigenvectors,
    uplo: u8,
) -> Result<EigenDecomposition<T, T::Real>, PackedMatrixError> {
    if n == 0 {
        return Ok(EigenDecomposition {
            eigenvalues: vec![],
            eigenvectors: (choice == Eigenvectors::Compute).then(Vec::new),
            dimension: 0,
        });
    }
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
    let mut work = vec![T::zero(); 1];
    let mut rw = vec![T::Real::zero(); 1];
    let mut iw = vec![0; 1];
    let mut info = 0;
    unsafe {
        T::pgvd(
            &[p.itype()],
            if compute { b'V' } else { b'N' },
            uplo,
            ni,
            &mut a,
            &mut b,
            &mut w,
            &mut z,
            ni.max(1),
            &mut work,
            -1,
            &mut rw,
            -1,
            &mut iw,
            -1,
            &mut info,
        )
    };
    generalized_info(n, info)?;
    let lw = T::work_len(work[0])
        .ok_or(PackedMatrixError::InvalidWorkspaceRecommendation { workspace: "WORK" })?;
    let lrw = if T::COMPLEX {
        T::real_work_len(rw[0])
            .ok_or(PackedMatrixError::InvalidWorkspaceRecommendation { workspace: "RWORK" })?
    } else {
        1
    };
    let liw = integer_workspace(iw[0], "IWORK")?;
    work.resize(lw, T::zero());
    rw.resize(lrw, T::Real::zero());
    iw.resize(liw, 0);
    unsafe {
        T::pgvd(
            &[p.itype()],
            if compute { b'V' } else { b'N' },
            uplo,
            ni,
            &mut a,
            &mut b,
            &mut w,
            &mut z,
            ni.max(1),
            &mut work,
            workspace_i32(lw, "WORK")?,
            &mut rw,
            workspace_i32(lrw, "RWORK")?,
            &mut iw,
            workspace_i32(liw, "IWORK")?,
            &mut info,
        )
    };
    generalized_info(n, info)?;
    Ok(EigenDecomposition {
        eigenvalues: w,
        eigenvectors: compute.then_some(z),
        dimension: n,
    })
}
fn generalized_selected<T: GeneralizedPackedEigen>(
    n: usize,
    mut a: Vec<T>,
    mut b: Vec<T>,
    p: GeneralizedEigenproblem,
    range: EigenRange<T::Real>,
    choice: Eigenvectors,
    uplo: u8,
) -> Result<SelectedEigenDecomposition<T, T::Real>, PackedMatrixError> {
    if n == 0 {
        return match range {
            EigenRange::Index { .. } => Err(PackedMatrixError::InvalidEigenRange {
                reason: "index range is invalid for an empty matrix",
            }),
            _ => Ok(SelectedEigenDecomposition {
                eigenvalues: vec![],
                eigenvectors: (choice == Eigenvectors::Compute).then(Vec::new),
                dimension: 0,
                count: 0,
            }),
        };
    }
    let ni = checked_n(n)?;
    let (r, vl, vu, il, iu) = selection(n, range, T::finite, T::less)?;
    let compute = choice == Eigenvectors::Compute;
    let mut m = 0;
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
    let mut work = vec![
        T::zero();
        n.checked_mul(if T::COMPLEX { 2 } else { 8 })
            .ok_or(PackedMatrixError::DimensionOverflow { n })?
    ];
    let mut rw = vec![
        T::Real::zero();
        if T::COMPLEX {
            n.checked_mul(7)
                .ok_or(PackedMatrixError::DimensionOverflow { n })?
        } else {
            1
        }
    ];
    let mut iw = vec![
        0;
        n.checked_mul(5)
            .ok_or(PackedMatrixError::DimensionOverflow { n })?
    ];
    let mut fail = vec![0; n];
    let mut info = 0;
    unsafe {
        T::pgvx(
            &[p.itype()],
            if compute { b'V' } else { b'N' },
            r,
            uplo,
            ni,
            &mut a,
            &mut b,
            vl,
            vu,
            il,
            iu,
            T::Real::zero(),
            &mut m,
            &mut w,
            &mut z,
            ni.max(1),
            &mut work,
            &mut rw,
            &mut iw,
            &mut fail,
            &mut info,
        )
    };
    if info > n as i32 {
        return Err(PackedMatrixError::PositiveDefinitenessFailure {
            index: (info - n as i32) as usize,
        });
    }
    selected_finish(n, m, w, compute.then_some(z), &fail, info)
}

macro_rules! generalized_methods {
    ($matrix:ident, $kind:path) => {
        impl<T: GeneralizedPackedEigen + $kind, S: PackedStorage<T>> $matrix<T, S> {
            pub fn generalized_eigendecomposition<B: PackedStorage<T>>(
                &self,
                b: &PackedSPD<T, B>,
                p: GeneralizedEigenproblem,
            ) -> Result<EigenDecomposition<T, T::Real>, PackedMatrixError> {
                if self.dimension() != b.dimension() {
                    return Err(PackedMatrixError::DimensionMismatch {
                        left: self.dimension(),
                        right: b.dimension(),
                    });
                }
                generalized_basic(
                    self.dimension(),
                    self.as_slice().to_vec(),
                    b.as_slice().to_vec(),
                    p,
                    Eigenvectors::Compute,
                    b'L',
                )
            }
            pub fn generalized_eigenvalues<B: PackedStorage<T>>(
                &self,
                b: &PackedSPD<T, B>,
                p: GeneralizedEigenproblem,
            ) -> Result<Vec<T::Real>, PackedMatrixError> {
                if self.dimension() != b.dimension() {
                    return Err(PackedMatrixError::DimensionMismatch {
                        left: self.dimension(),
                        right: b.dimension(),
                    });
                }
                Ok(generalized_basic(
                    self.dimension(),
                    self.as_slice().to_vec(),
                    b.as_slice().to_vec(),
                    p,
                    Eigenvectors::None,
                    b'L',
                )?
                .eigenvalues)
            }
            pub fn generalized_eigendecomposition_divide_conquer<B: PackedStorage<T>>(
                &self,
                b: &PackedSPD<T, B>,
                p: GeneralizedEigenproblem,
            ) -> Result<EigenDecomposition<T, T::Real>, PackedMatrixError> {
                if self.dimension() != b.dimension() {
                    return Err(PackedMatrixError::DimensionMismatch {
                        left: self.dimension(),
                        right: b.dimension(),
                    });
                }
                generalized_dc(
                    self.dimension(),
                    self.as_slice().to_vec(),
                    b.as_slice().to_vec(),
                    p,
                    Eigenvectors::Compute,
                    b'L',
                )
            }
            pub fn generalized_eigenvalues_divide_conquer<B: PackedStorage<T>>(
                &self,
                b: &PackedSPD<T, B>,
                p: GeneralizedEigenproblem,
            ) -> Result<Vec<T::Real>, PackedMatrixError> {
                if self.dimension() != b.dimension() {
                    return Err(PackedMatrixError::DimensionMismatch {
                        left: self.dimension(),
                        right: b.dimension(),
                    });
                }
                Ok(generalized_dc(
                    self.dimension(),
                    self.as_slice().to_vec(),
                    b.as_slice().to_vec(),
                    p,
                    Eigenvectors::None,
                    b'L',
                )?
                .eigenvalues)
            }
            pub fn generalized_selected_eigendecomposition<B: PackedStorage<T>>(
                &self,
                b: &PackedSPD<T, B>,
                p: GeneralizedEigenproblem,
                r: EigenRange<T::Real>,
            ) -> Result<SelectedEigenDecomposition<T, T::Real>, PackedMatrixError> {
                if self.dimension() != b.dimension() {
                    return Err(PackedMatrixError::DimensionMismatch {
                        left: self.dimension(),
                        right: b.dimension(),
                    });
                }
                generalized_selected(
                    self.dimension(),
                    self.as_slice().to_vec(),
                    b.as_slice().to_vec(),
                    p,
                    r,
                    Eigenvectors::Compute,
                    b'L',
                )
            }
            pub fn generalized_selected_eigenvalues<B: PackedStorage<T>>(
                &self,
                b: &PackedSPD<T, B>,
                p: GeneralizedEigenproblem,
                r: EigenRange<T::Real>,
            ) -> Result<Vec<T::Real>, PackedMatrixError> {
                if self.dimension() != b.dimension() {
                    return Err(PackedMatrixError::DimensionMismatch {
                        left: self.dimension(),
                        right: b.dimension(),
                    });
                }
                Ok(generalized_selected(
                    self.dimension(),
                    self.as_slice().to_vec(),
                    b.as_slice().to_vec(),
                    p,
                    r,
                    Eigenvectors::None,
                    b'L',
                )?
                .eigenvalues)
            }
        }
    };
}
generalized_methods!(PackedSymmetric, SymmetricPackedSelected);
generalized_methods!(PackedHermitian, HermitianPackedSelected);

macro_rules! generalized_consuming_methods {
    ($matrix:ident, $kind:path) => {
        impl<T: GeneralizedPackedEigen + $kind> $matrix<T, Vec<T>> {
            /// Consumes both operands and reuses their packed allocations.
            pub fn into_generalized_eigendecomposition(
                self,
                b: PackedSPD<T>,
                p: GeneralizedEigenproblem,
            ) -> Result<EigenDecomposition<T, T::Real>, PackedMatrixError> {
                let n = self.dimension();
                if n != b.dimension() {
                    return Err(PackedMatrixError::DimensionMismatch {
                        left: n,
                        right: b.dimension(),
                    });
                }
                generalized_basic(
                    n,
                    self.into_vec(),
                    b.into_vec(),
                    p,
                    Eigenvectors::Compute,
                    b'L',
                )
            }
            /// Consumes both operands for the divide-and-conquer driver.
            pub fn into_generalized_eigendecomposition_divide_conquer(
                self,
                b: PackedSPD<T>,
                p: GeneralizedEigenproblem,
            ) -> Result<EigenDecomposition<T, T::Real>, PackedMatrixError> {
                let n = self.dimension();
                if n != b.dimension() {
                    return Err(PackedMatrixError::DimensionMismatch {
                        left: n,
                        right: b.dimension(),
                    });
                }
                generalized_dc(
                    n,
                    self.into_vec(),
                    b.into_vec(),
                    p,
                    Eigenvectors::Compute,
                    b'L',
                )
            }
        }
    };
}
generalized_consuming_methods!(PackedSymmetric, SymmetricPackedSelected);
generalized_consuming_methods!(PackedHermitian, HermitianPackedSelected);

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
    #[test]
    fn selected_real_ranges() {
        let a = PackedSymmetric::from_vec(3, vec![1_f64, 0., 0., 2., 0., 3.]).unwrap();
        assert_eq!(
            a.selected_eigenvalues(EigenRange::All).unwrap(),
            vec![1., 2., 3.]
        );
        assert_eq!(
            a.selected_eigenvalues(EigenRange::Index { first: 1, last: 2 })
                .unwrap(),
            vec![2., 3.]
        );
        assert_eq!(
            a.selected_eigenvalues(EigenRange::Value {
                lower: 1.,
                upper: 2.
            })
            .unwrap(),
            vec![2.]
        );
        assert_eq!(
            a.selected_eigenvalues(EigenRange::Value {
                lower: 4.,
                upper: 5.
            })
            .unwrap(),
            vec![]
        );
        assert_eq!(
            a.selected_eigendecomposition(EigenRange::Index { first: 0, last: 1 })
                .unwrap()
                .eigenvectors
                .unwrap()
                .len(),
            6
        );
    }
    #[test]
    fn selected_complex_and_validation() {
        let c = Complex64::new;
        let a = PackedHermitian::from_vec(2, vec![c(2., 0.), c(0., 1.), c(2., 0.)]).unwrap();
        close(
            a.selected_eigenvalues(EigenRange::Index { first: 0, last: 0 })
                .unwrap()[0],
            1.,
        );
        assert!(
            a.selected_eigenvalues(EigenRange::Index { first: 1, last: 0 })
                .is_err()
        );
        assert!(
            a.selected_eigenvalues(EigenRange::Index { first: 0, last: 2 })
                .is_err()
        );
        assert!(
            a.selected_eigenvalues(EigenRange::Value {
                lower: f64::NAN,
                upper: 2.
            })
            .is_err()
        );
        assert!(
            a.selected_eigenvalues(EigenRange::Value {
                lower: 2.,
                upper: 2.
            })
            .is_err()
        );
    }
    #[test]
    fn selected_f32_upper_lower_and_empty() {
        let a = PackedSymmetric::from_vec(2, vec![2_f32, 1., 2.]).unwrap();
        assert_eq!(a.selected_eigenvalues(EigenRange::All).unwrap().len(), 2);
        for u in [b'L', b'U'] {
            assert_eq!(
                symmetric_selected(
                    2,
                    vec![2_f64, 1., 2.],
                    EigenRange::All,
                    Eigenvectors::Compute,
                    0.,
                    u
                )
                .unwrap()
                .count,
                2
            )
        }
        let e = PackedHermitian::<Complex32>::from_vec(0, vec![]).unwrap();
        assert_eq!(e.selected_eigenvalues(EigenRange::All).unwrap(), vec![]);
    }
    #[test]
    fn generalized_real_algorithms_and_problems() {
        let a = PackedSymmetric::from_vec(2, vec![4_f64, 0., 9.]).unwrap();
        let b = PackedSPD::from_vec(2, vec![2_f64, 0., 3.]).unwrap();
        let e = a
            .generalized_eigendecomposition(&b, GeneralizedEigenproblem::AxEqualsLambdaBx)
            .unwrap();
        close(e.eigenvalues[0], 2.);
        close(e.eigenvalues[1], 3.);
        for p in [
            GeneralizedEigenproblem::ABxEqualsLambdaX,
            GeneralizedEigenproblem::BAxEqualsLambdaX,
        ] {
            let v = a.generalized_eigenvalues(&b, p).unwrap();
            close(v[0], 8.);
            close(v[1], 27.)
        }
        let dc = a
            .generalized_eigendecomposition_divide_conquer(
                &b,
                GeneralizedEigenproblem::AxEqualsLambdaBx,
            )
            .unwrap();
        close(dc.eigenvalues[0], 2.);
        let s = a
            .generalized_selected_eigendecomposition(
                &b,
                GeneralizedEigenproblem::AxEqualsLambdaBx,
                EigenRange::Index { first: 1, last: 1 },
            )
            .unwrap();
        assert_eq!(s.count, 1);
        close(s.eigenvalues[0], 3.);
    }
    #[test]
    fn generalized_complex_f32_and_errors() {
        let c = Complex32::new;
        let a = PackedHermitian::from_vec(2, vec![c(4., 0.), c(0., 0.), c(9., 0.)]).unwrap();
        let b = PackedSPD::from_vec(2, vec![c(2., 0.), c(0., 0.), c(3., 0.)]).unwrap();
        let values = a
            .generalized_eigenvalues(&b, GeneralizedEigenproblem::AxEqualsLambdaBx)
            .unwrap();
        assert!((values[0] - 2.).abs() < 1e-5 && (values[1] - 3.).abs() < 1e-5);
        let short = PackedSPD::from_vec(1, vec![c(1., 0.)]).unwrap();
        assert!(matches!(
            a.generalized_eigenvalues(&short, GeneralizedEigenproblem::AxEqualsLambdaBx),
            Err(PackedMatrixError::DimensionMismatch { .. })
        ));
        let bad = PackedSPD::from_vec(2, vec![c(-1., 0.), c(0., 0.), c(1., 0.)]).unwrap();
        assert!(matches!(
            a.generalized_eigenvalues(&bad, GeneralizedEigenproblem::AxEqualsLambdaBx),
            Err(PackedMatrixError::PositiveDefinitenessFailure { index: 1 })
        ));
    }
    #[test]
    fn generalized_real_f32_selected_value() {
        let a = PackedSymmetric::from_vec(2, vec![4_f32, 0., 9.]).unwrap();
        let b = PackedSPD::from_vec(2, vec![2_f32, 0., 3.]).unwrap();
        assert_eq!(
            a.generalized_selected_eigenvalues(
                &b,
                GeneralizedEigenproblem::AxEqualsLambdaBx,
                EigenRange::Value {
                    lower: 2.5,
                    upper: 3.
                }
            )
            .unwrap(),
            vec![3.]
        );
    }
    #[test]
    fn generalized_followup_coverage() {
        let a = PackedSymmetric::from_vec(2, vec![4_f64, 0., 9.]).unwrap();
        let b = PackedSPD::from_vec(2, vec![2_f64, 0., 3.]).unwrap();
        let values = a
            .generalized_eigenvalues_divide_conquer(&b, GeneralizedEigenproblem::AxEqualsLambdaBx)
            .unwrap();
        close(values[0], 2.);
        close(values[1], 3.);
        let consumed = a
            .clone()
            .into_generalized_eigendecomposition(
                b.clone(),
                GeneralizedEigenproblem::AxEqualsLambdaBx,
            )
            .unwrap();
        close(consumed.eigenvalues[0], 2.);
        for u in [b'L', b'U'] {
            let e = generalized_basic(
                2,
                vec![4_f64, 0., 9.],
                vec![2_f64, 0., 3.],
                GeneralizedEigenproblem::AxEqualsLambdaBx,
                Eigenvectors::Compute,
                u,
            )
            .unwrap();
            close(e.eigenvalues[1], 3.);
        }
        let c = Complex64::new;
        let h = PackedHermitian::from_vec(2, vec![c(4., 0.), c(0., 0.), c(9., 0.)]).unwrap();
        let hp = PackedSPD::from_vec(2, vec![c(2., 0.), c(0., 0.), c(3., 0.)]).unwrap();
        let selected = h
            .generalized_selected_eigenvalues(
                &hp,
                GeneralizedEigenproblem::AxEqualsLambdaBx,
                EigenRange::Value {
                    lower: 2.5,
                    upper: 3.5,
                },
            )
            .unwrap();
        assert_eq!(selected.len(), 1);
        close(selected[0], 3.);
    }
}
