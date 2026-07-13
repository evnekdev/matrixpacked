use crate::{
    PackedHermitian, PackedMatrixError, PackedSPD, PackedSymmetric,
    backend::{
        HermitianPackedSolveDriver, PositiveDefinitePackedSolveDriver, SymmetricPackedSolveDriver,
    },
    factorization::{check_info, check_rhs_many, checked_n},
    storage::{PackedStorage, PackedStorageMut},
};

fn driver_dimensions<T>(
    n: usize,
    nrhs: usize,
    rhs: &[T],
) -> Result<(i32, i32, i32), PackedMatrixError> {
    check_rhs_many(n, nrhs, rhs)?;
    let lapack_n = checked_n(n)?;
    Ok((lapack_n, checked_n(nrhs)?, lapack_n.max(1)))
}

fn ppsv_in_place<T: PositiveDefinitePackedSolveDriver>(
    n: usize,
    ap: &mut [T],
    rhs: &mut [T],
    nrhs: usize,
) -> Result<(), PackedMatrixError> {
    let (n, nrhs, ldb) = driver_dimensions(n, nrhs, rhs)?;
    let mut info = 0;
    unsafe { T::ppsv(b'L', n, nrhs, ap, rhs, ldb, &mut info) };
    check_info(info, "matrix is not positive definite")
}

fn spsv_in_place<T: SymmetricPackedSolveDriver>(
    n: usize,
    ap: &mut [T],
    rhs: &mut [T],
    nrhs: usize,
) -> Result<(), PackedMatrixError> {
    let (lapack_n, nrhs, ldb) = driver_dimensions(n, nrhs, rhs)?;
    let mut pivots = vec![0; n];
    let mut info = 0;
    unsafe { T::spsv(b'L', lapack_n, nrhs, ap, &mut pivots, rhs, ldb, &mut info) };
    check_info(info, "symmetric packed matrix is singular")
}

fn hpsv_in_place<T: HermitianPackedSolveDriver>(
    n: usize,
    ap: &mut [T],
    rhs: &mut [T],
    nrhs: usize,
) -> Result<(), PackedMatrixError> {
    let (lapack_n, nrhs, ldb) = driver_dimensions(n, nrhs, rhs)?;
    let mut pivots = vec![0; n];
    let mut info = 0;
    unsafe { T::hpsv(b'L', lapack_n, nrhs, ap, &mut pivots, rhs, ldb, &mut info) };
    check_info(info, "Hermitian packed matrix is singular")
}

macro_rules! impl_simple_solve {
    ($matrix:ident, $driver:ident, $helper:ident) => {
        impl<T, S> $matrix<T, S>
        where
            T: $driver,
            S: PackedStorage<T>,
        {
            /// Factors a packed copy and returns solutions in column-major RHS layout.
            ///
            /// The original matrix and right-hand sides are unchanged. Retain a reusable
            /// factorization instead when solving more than once with the same matrix.
            pub fn solve_once(&self, rhs: &[T], nrhs: usize) -> Result<Vec<T>, PackedMatrixError> {
                check_rhs_many(self.dimension(), nrhs, rhs)?;
                let mut packed = self.as_slice().to_vec();
                let mut solution = rhs.to_vec();
                $helper(self.dimension(), &mut packed, &mut solution, nrhs)?;
                Ok(solution)
            }
        }

        impl<T, S> $matrix<T, S>
        where
            T: $driver,
            S: PackedStorageMut<T>,
        {
            /// Factors this matrix and solves column-major right-hand sides in place.
            ///
            /// The packed matrix is overwritten by LAPACK's factorization even when the
            /// routine later reports a singular or non-positive-definite matrix.
            pub fn solve_once_in_place(
                &mut self,
                rhs: &mut [T],
                nrhs: usize,
            ) -> Result<(), PackedMatrixError> {
                let n = self.dimension();
                $helper(n, self.as_mut_slice(), rhs, nrhs)
            }
        }

        impl<T> $matrix<T, Vec<T>>
        where
            T: $driver,
        {
            /// Consumes and reuses owned packed storage for a one-shot solve.
            pub fn into_solve_once(
                mut self,
                mut rhs: Vec<T>,
                nrhs: usize,
            ) -> Result<Vec<T>, PackedMatrixError> {
                self.solve_once_in_place(&mut rhs, nrhs)?;
                Ok(rhs)
            }
        }
    };
}

impl_simple_solve!(PackedSPD, PositiveDefinitePackedSolveDriver, ppsv_in_place);
impl_simple_solve!(PackedSymmetric, SymmetricPackedSolveDriver, spsv_in_place);
impl_simple_solve!(PackedHermitian, HermitianPackedSolveDriver, hpsv_in_place);

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn positive_definite_borrowing_consuming_and_multiple_rhs() {
        let a = PackedSPD::from_vec(2, vec![4.0f64, 1.0, 3.0]).unwrap();
        let rhs = [6.0, 7.0, 1.0, 0.0];
        let solution = a.solve_once(&rhs, 2).unwrap();
        assert_close(&solution, &[1.0, 2.0, 3.0 / 11.0, -1.0 / 11.0]);
        assert_eq!(a.as_slice(), &[4.0, 1.0, 3.0]);
        assert_eq!(rhs, [6.0, 7.0, 1.0, 0.0]);

        let consumed = a.into_solve_once(rhs.to_vec(), 2).unwrap();
        assert_close(&consumed, &solution);
    }

    #[test]
    fn mutable_view_validation_zero_rhs_and_spd_failure() {
        let mut storage = [4.0f32, 1.0, 3.0];
        let mut view = PackedSPD::from_slice_mut(2, &mut storage).unwrap();
        let mut rhs = [6.0f32, 7.0];
        view.solve_once_in_place(&mut rhs, 1).unwrap();
        assert!((rhs[0] - 1.0).abs() < 1e-5 && (rhs[1] - 2.0).abs() < 1e-5);

        let empty = PackedSPD::from_vec(2, vec![4.0f32, 1.0, 3.0]).unwrap();
        assert!(empty.solve_once(&[], 0).unwrap().is_empty());
        assert_eq!(
            empty.solve_once(&[1.0], 1),
            Err(PackedMatrixError::InvalidVectorLength {
                expected: 2,
                actual: 1
            })
        );
        let not_pd = PackedSPD::from_vec(2, vec![1.0f32, 0.0, -1.0]).unwrap();
        assert!(matches!(
            not_pd.solve_once(&[1.0, 1.0], 1),
            Err(PackedMatrixError::FactorizationFailure { index: 2, .. })
        ));
    }

    #[test]
    fn symmetric_real_and_complex_match_reusable_factorization() {
        let real = PackedSymmetric::from_vec(2, vec![0.0f64, 1.0, 0.0]).unwrap();
        let rhs = [2.0, 3.0];
        assert_close(
            &real.solve_once(&rhs, 1).unwrap(),
            &real.factorize().unwrap().solve_vector(&rhs).unwrap(),
        );

        let c = Complex32::new;
        let complex =
            PackedSymmetric::from_vec(2, vec![c(0.0, 0.0), c(1.0, 1.0), c(0.0, 0.0)]).unwrap();
        let rhs = [c(2.0, 0.0), c(0.0, 1.0)];
        let once = complex.solve_once(&rhs, 1).unwrap();
        let reusable = complex.factorize().unwrap().solve_vector(&rhs).unwrap();
        for (actual, expected) in once.iter().zip(reusable) {
            assert!((*actual - expected).norm() < 1e-5);
        }

        let singular = PackedSymmetric::from_vec(2, vec![1.0f64, 0.0, 0.0]).unwrap();
        assert!(matches!(
            singular.solve_once(&rhs.map(|z| z.re as f64), 1),
            Err(PackedMatrixError::FactorizationFailure { .. })
        ));
    }

    #[test]
    fn hermitian_complex_and_singular() {
        let c = Complex64::new;
        let matrix =
            PackedHermitian::from_vec(2, vec![c(0.0, 0.0), c(1.0, -1.0), c(0.0, 0.0)]).unwrap();
        let rhs = [c(2.0, 0.0), c(0.0, 1.0)];
        let once = matrix.solve_once(&rhs, 1).unwrap();
        let reusable = matrix.factorize().unwrap().solve_vector(&rhs).unwrap();
        for (actual, expected) in once.iter().zip(reusable) {
            assert!((*actual - expected).norm() < 1e-12);
        }

        let singular =
            PackedHermitian::from_vec(2, vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]).unwrap();
        assert!(matches!(
            singular.solve_once(&rhs, 1),
            Err(PackedMatrixError::FactorizationFailure { .. })
        ));
    }
}
