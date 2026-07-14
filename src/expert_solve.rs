use num_traits::Zero;

use crate::{
    PackedHermitian, PackedMatrixError, PackedSPD, PackedSymmetric,
    backend::{
        HermitianPackedExpertDriver, IndefinitePackedExpertDriver,
        PositiveDefinitePackedExpertDriver,
    },
    factorization::{check_rhs_many, checked_n, checked_workspace_len},
    storage::PackedStorage,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Controls whether the positive-definite expert driver equilibrates the system.
pub enum EquilibrationMode {
    /// Solve the original system without equilibration.
    #[default]
    None,
    /// Let `xPPSVX` compute and apply diagonal scaling when beneficial.
    Compute,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Options for positive-definite expert solves.
pub struct ExpertSolveOptions {
    /// Requested equilibration behavior.
    pub equilibration: EquilibrationMode,
}

#[derive(Clone, Debug, PartialEq)]
/// Solution and diagnostics returned by LAPACK expert drivers.
///
/// Solutions are column-major: `nrhs` consecutive columns of length `n`.
/// Error vectors contain one entry per RHS.
pub struct ExpertSolveResult<T, R> {
    /// Column-major solution buffer.
    pub solution: Vec<T>,
    /// Estimate of `1 / cond(A)` in the one-norm.
    pub reciprocal_condition_number: R,
    /// Forward error estimate (`ferr`) for each RHS.
    pub forward_error: Vec<R>,
    /// Componentwise backward error (`berr`) for each RHS.
    pub backward_error: Vec<R>,
    /// Scaling factors actually applied by `xPPSVX`; absent when LAPACK did not equilibrate.
    pub equilibration: Option<Vec<R>>,
}

fn dimensions<T>(n: usize, nrhs: usize, rhs: &[T]) -> Result<(i32, i32, i32), PackedMatrixError> {
    check_rhs_many(n, nrhs, rhs)?;
    let n = checked_n(n)?;
    Ok((n, checked_n(nrhs)?, n.max(1)))
}

fn expert_info(info: i32, n: i32, message: &'static str) -> Result<(), PackedMatrixError> {
    if info < 0 {
        Err(PackedMatrixError::LapackIllegalArgument { argument: -info })
    } else if info > 0 && info <= n {
        Err(PackedMatrixError::FactorizationFailure {
            index: info as usize,
            message,
        })
    } else {
        Ok(())
    }
}

impl<T, S> PackedSPD<T, S>
where
    T: PositiveDefinitePackedExpertDriver,
    T::Real: Zero,
    S: PackedStorage<T>,
{
    /// Solves an SPD/HPD system with factorization, condition estimation, and refinement.
    ///
    /// This convenience form disables equilibration. `rhs` contains `nrhs`
    /// column-major columns. Supplied-factor mode is not exposed; the expert
    /// driver computes a fresh factorization of an internal copy.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid RHS layout, dimension/workspace overflow,
    /// a non-positive-definite matrix, or an illegal LAPACK argument.
    pub fn expert_solve(
        &self,
        rhs: &[T],
        nrhs: usize,
    ) -> Result<ExpertSolveResult<T, T::Real>, PackedMatrixError> {
        self.expert_solve_with_options(rhs, nrhs, ExpertSolveOptions::default())
    }
    /// Runs the positive-definite expert driver with explicit equilibration options.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use matrixpacked::{EquilibrationMode, ExpertSolveOptions, PackedSPD};
    /// let a = PackedSPD::from_vec(2, vec![4.0_f64, 1.0, 3.0])?;
    /// let result = a.expert_solve_with_options(
    ///     &[6.0, 7.0],
    ///     1,
    ///     ExpertSolveOptions { equilibration: EquilibrationMode::Compute },
    /// )?;
    /// assert_eq!(result.forward_error.len(), 1);
    /// # Ok::<(), matrixpacked::PackedMatrixError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error for invalid RHS layout, dimension/workspace overflow,
    /// a non-positive-definite matrix, or an illegal LAPACK argument.
    pub fn expert_solve_with_options(
        &self,
        rhs: &[T],
        nrhs: usize,
        options: ExpertSolveOptions,
    ) -> Result<ExpertSolveResult<T, T::Real>, PackedMatrixError> {
        let (n, nrhs, ld) = dimensions(self.dimension(), nrhs, rhs)?;
        let mut ap = self.as_slice().to_vec();
        let mut afp = vec![T::zero(); ap.len()];
        let mut b = rhs.to_vec();
        let mut x = vec![T::zero(); rhs.len()];
        let mut scaling = vec![T::Real::zero(); self.dimension()];
        let mut equed = b'N';
        let mut rcond = T::Real::zero();
        let count = nrhs as usize;
        let mut ferr = vec![T::Real::zero(); count];
        let mut berr = vec![T::Real::zero(); count];
        let mut work =
            vec![
                T::zero();
                checked_workspace_len(self.dimension(), if T::EXPERT_IS_COMPLEX { 2 } else { 3 })?
            ];
        let mut realwork = vec![
            T::Real::zero();
            if T::EXPERT_IS_COMPLEX {
                self.dimension()
            } else {
                0
            }
        ];
        let mut iwork = vec![
            0;
            if T::EXPERT_IS_COMPLEX {
                0
            } else {
                self.dimension()
            }
        ];
        let mut info = 0;
        let fact = match options.equilibration {
            EquilibrationMode::None => b'N',
            EquilibrationMode::Compute => b'E',
        };
        unsafe {
            T::ppsvx(
                fact,
                b'L',
                n,
                nrhs,
                &mut ap,
                &mut afp,
                &mut equed,
                &mut scaling,
                &mut b,
                ld,
                &mut x,
                ld,
                &mut rcond,
                &mut ferr,
                &mut berr,
                &mut work,
                &mut realwork,
                &mut iwork,
                &mut info,
            )
        };
        expert_info(info, n, "matrix is not positive definite")?;
        Ok(ExpertSolveResult {
            solution: x,
            reciprocal_condition_number: rcond,
            forward_error: ferr,
            backward_error: berr,
            equilibration: (equed == b'Y').then_some(scaling),
        })
    }
}

fn symmetric_expert<T: IndefinitePackedExpertDriver>(
    n: usize,
    ap: &[T],
    rhs: &[T],
    nrhs: usize,
) -> Result<ExpertSolveResult<T, T::Real>, PackedMatrixError>
where
    T::Real: Zero,
{
    let (ln, lr, ld) = dimensions(n, nrhs, rhs)?;
    let mut afp = vec![T::zero(); ap.len()];
    let mut ipiv = vec![0; n];
    let mut x = vec![T::zero(); rhs.len()];
    let mut rcond = T::Real::zero();
    let mut ferr = vec![T::Real::zero(); nrhs];
    let mut berr = vec![T::Real::zero(); nrhs];
    let mut work =
        vec![T::zero(); checked_workspace_len(n, if T::EXPERT_IS_COMPLEX { 2 } else { 3 })?];
    let mut rw = vec![T::Real::zero(); if T::EXPERT_IS_COMPLEX { n } else { 0 }];
    let mut iw = vec![0; if T::EXPERT_IS_COMPLEX { 0 } else { n }];
    let mut info = 0;
    unsafe {
        T::packed_svx(
            b'N', b'L', ln, lr, ap, &mut afp, &mut ipiv, rhs, ld, &mut x, ld, &mut rcond,
            &mut ferr, &mut berr, &mut work, &mut rw, &mut iw, &mut info,
        )
    };
    expert_info(info, ln, "symmetric packed matrix is singular")?;
    Ok(ExpertSolveResult {
        solution: x,
        reciprocal_condition_number: rcond,
        forward_error: ferr,
        backward_error: berr,
        equilibration: None,
    })
}

impl<T, S> PackedSymmetric<T, S>
where
    T: IndefinitePackedExpertDriver,
    T::Real: Zero,
    S: PackedStorage<T>,
{
    /// Solves a symmetric-indefinite system with condition/error estimates.
    ///
    /// `rhs` contains `nrhs` column-major columns. The driver computes a fresh
    /// pivoted factorization; supplied-factor and equilibration modes are not exposed.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid RHS layout, dimension/workspace overflow,
    /// singular factorization, or illegal LAPACK arguments.
    pub fn expert_solve(
        &self,
        rhs: &[T],
        nrhs: usize,
    ) -> Result<ExpertSolveResult<T, T::Real>, PackedMatrixError> {
        symmetric_expert(self.dimension(), self.as_slice(), rhs, nrhs)
    }
}

impl<T, S> PackedHermitian<T, S>
where
    T: HermitianPackedExpertDriver,
    T::Real: Zero,
    S: PackedStorage<T>,
{
    /// Solves a Hermitian-indefinite system with condition/error estimates.
    ///
    /// `rhs` contains `nrhs` column-major columns. The driver computes a fresh
    /// pivoted factorization; supplied-factor and equilibration modes are not exposed.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid RHS layout, dimension/workspace overflow,
    /// singular factorization, or illegal LAPACK arguments.
    pub fn expert_solve(
        &self,
        rhs: &[T],
        nrhs: usize,
    ) -> Result<ExpertSolveResult<T, T::Real>, PackedMatrixError> {
        let n = self.dimension();
        let (ln, lr, ld) = dimensions(n, nrhs, rhs)?;
        let mut afp = vec![T::zero(); self.as_slice().len()];
        let mut ipiv = vec![0; n];
        let mut x = vec![T::zero(); rhs.len()];
        let mut rcond = T::Real::zero();
        let mut ferr = vec![T::Real::zero(); nrhs];
        let mut berr = vec![T::Real::zero(); nrhs];
        let mut work = vec![T::zero(); checked_workspace_len(n, 2)?];
        let mut rw = vec![T::Real::zero(); n];
        let mut info = 0;
        unsafe {
            T::hpsvx(
                b'N',
                b'L',
                ln,
                lr,
                self.as_slice(),
                &mut afp,
                &mut ipiv,
                rhs,
                ld,
                &mut x,
                ld,
                &mut rcond,
                &mut ferr,
                &mut berr,
                &mut work,
                &mut rw,
                &mut info,
            )
        };
        expert_info(info, ln, "Hermitian packed matrix is singular")?;
        Ok(ExpertSolveResult {
            solution: x,
            reciprocal_condition_number: rcond,
            forward_error: ferr,
            backward_error: berr,
            equilibration: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    #[test]
    fn spd_options_and_lengths() {
        let a = PackedSPD::from_vec(2, vec![1e-8f64, 0.0, 1e8]).unwrap();
        let r = a
            .expert_solve_with_options(
                &[1.0, 1.0],
                1,
                ExpertSolveOptions {
                    equilibration: EquilibrationMode::Compute,
                },
            )
            .unwrap();
        assert_eq!(r.solution.len(), 2);
        assert_eq!(r.forward_error.len(), 1);
        assert!(r.reciprocal_condition_number >= 0.0);
        assert!(r.equilibration.is_some());
        assert_eq!(a.as_slice(), &[1e-8, 0.0, 1e8]);
    }
    #[test]
    fn indefinite_real_complex_and_validation() {
        let a = PackedSymmetric::from_vec(2, vec![0.0f64, 1.0, 0.0]).unwrap();
        let r = a.expert_solve(&[2.0, 3.0], 1).unwrap();
        assert!((r.solution[0] - 3.0).abs() < 1e-12);
        assert!(a.expert_solve(&[1.0], 1).is_err());
        let c = Complex64::new;
        let h = PackedHermitian::from_vec(2, vec![c(0.0, 0.0), c(1.0, -1.0), c(0.0, 0.0)]).unwrap();
        assert_eq!(
            h.expert_solve(&[c(1.0, 0.0), c(0.0, 1.0)], 1)
                .unwrap()
                .solution
                .len(),
            2
        );
    }

    #[test]
    fn multiple_rhs_and_factorization_failures() {
        let a = PackedSPD::from_vec(2, vec![4.0f64, 1.0, 3.0]).unwrap();
        let result = a.expert_solve(&[6.0, 7.0, 1.0, 0.0], 2).unwrap();
        assert_eq!(result.solution.len(), 4);
        assert_eq!(result.forward_error.len(), 2);
        assert_eq!(result.backward_error.len(), 2);

        let not_pd = PackedSPD::from_vec(2, vec![1.0f64, 0.0, -1.0]).unwrap();
        assert!(matches!(
            not_pd.expert_solve(&[1.0, 1.0], 1),
            Err(PackedMatrixError::FactorizationFailure { .. })
        ));
        let singular = PackedSymmetric::from_vec(2, vec![1.0f64, 0.0, 0.0]).unwrap();
        assert!(matches!(
            singular.expert_solve(&[1.0, 1.0], 1),
            Err(PackedMatrixError::FactorizationFailure { .. })
        ));
        let c = Complex64::new;
        let singular_h =
            PackedHermitian::from_vec(2, vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]).unwrap();
        assert!(matches!(
            singular_h.expert_solve(&[c(1.0, 0.0), c(1.0, 0.0)], 1),
            Err(PackedMatrixError::FactorizationFailure { .. })
        ));
    }
}
