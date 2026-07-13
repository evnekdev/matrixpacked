use crate::{
    GeneralizedEigenproblem, PackedHermitian, PackedMatrixError, PackedSymmetric,
    backend::GeneralizedPackedReduction,
    factorization::{PackedCholesky, check_info, checked_n},
    storage::{PackedStorage, PackedStorageMut},
};

fn reduce<T, B>(
    n: usize,
    data: &mut [T],
    b: &PackedCholesky<T, B>,
    problem: GeneralizedEigenproblem,
) -> Result<(), PackedMatrixError>
where
    T: GeneralizedPackedReduction,
    B: PackedStorage<T>,
{
    if n != b.dimension() {
        return Err(PackedMatrixError::DimensionMismatch {
            left: n,
            right: b.dimension(),
        });
    }
    let mut info = 0;
    unsafe {
        T::pgst(
            &[problem.itype()],
            b'L',
            checked_n(n)?,
            data,
            b.as_slice(),
            &mut info,
        )
    };
    check_info(info, "generalized packed reduction failed")
}

macro_rules! impl_reduction {
    ($matrix:ident) => {
        impl<T, S> $matrix<T, S>
        where
            T: GeneralizedPackedReduction,
            S: PackedStorage<T>,
        {
            /// Reduces a generalized-definite problem to standard form using the packed Cholesky factor `B = L L^H`.
            ///
            /// For [`GeneralizedEigenproblem::AxEqualsLambdaBx`], the result is
            /// `C = L^-1 A L^-H`. For the other two variants it is
            /// `C = L^H A L`. After solving `C y = lambda y`, recover an original
            /// eigenvector with `x = L^-H y` for `A x = lambda B x` and
            /// `A B x = lambda x`, or with `x = L y` for `B A x = lambda x`.
            /// Here `H` means transpose for real matrices and conjugate transpose
            /// for complex matrices. The borrowed factor `b` is not modified.
            pub fn generalized_reduction<B: PackedStorage<T>>(
                &self,
                b: &PackedCholesky<T, B>,
                problem: GeneralizedEigenproblem,
            ) -> Result<$matrix<T>, PackedMatrixError> {
                let mut out = $matrix::from_vec(self.dimension(), self.as_slice().to_vec())?;
                out.reduce_generalized_in_place(b, problem)?;
                Ok(out)
            }
        }
        impl<T, S> $matrix<T, S>
        where
            T: GeneralizedPackedReduction,
            S: PackedStorageMut<T>,
        {
            /// Overwrites this packed matrix with the corresponding standard-form matrix.
            ///
            /// See [`Self::generalized_reduction`] for the three transformations
            /// and the required eigenvector back-transform. The borrowed factor
            /// `b` is unchanged.
            pub fn reduce_generalized_in_place<B: PackedStorage<T>>(
                &mut self,
                b: &PackedCholesky<T, B>,
                problem: GeneralizedEigenproblem,
            ) -> Result<(), PackedMatrixError> {
                let n = self.dimension();
                reduce(n, self.as_mut_slice(), b, problem)
            }
        }
        impl<T> $matrix<T, Vec<T>>
        where
            T: GeneralizedPackedReduction,
        {
            /// Consumes and reuses owned packed `A` storage for generalized reduction.
            ///
            /// See [`Self::generalized_reduction`] for the mathematical conventions.
            pub fn into_generalized_reduction<B: PackedStorage<T>>(
                mut self,
                b: &PackedCholesky<T, B>,
                problem: GeneralizedEigenproblem,
            ) -> Result<Self, PackedMatrixError> {
                self.reduce_generalized_in_place(b, problem)?;
                Ok(self)
            }
        }
    };
}
impl_reduction!(PackedSymmetric);
impl_reduction!(PackedHermitian);

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    const PROBLEMS: [GeneralizedEigenproblem; 3] = [
        GeneralizedEigenproblem::AxEqualsLambdaBx,
        GeneralizedEigenproblem::ABxEqualsLambdaX,
        GeneralizedEigenproblem::BAxEqualsLambdaX,
    ];
    #[test]
    fn real_all_problem_types_and_mismatch() {
        let a = PackedSymmetric::from_vec(2, vec![3f64, 1., 2.]).unwrap();
        let b = crate::PackedSPD::from_vec(2, vec![2f64, 0.25, 1.5]).unwrap();
        let bf = b.cholesky().unwrap();
        let factor_before = bf.as_slice().to_vec();
        for p in PROBLEMS {
            let expected = a.generalized_eigenvalues(&b, p).unwrap();
            let reduced = a.generalized_reduction(&bf, p).unwrap();
            let actual = reduced.eigenvalues().unwrap();
            for (x, y) in actual.iter().zip(expected) {
                assert!((*x - y).abs() < 1e-10);
            }
        }
        assert_eq!(bf.as_slice(), factor_before);
        let short = crate::PackedSPD::from_vec(1, vec![1f64])
            .unwrap()
            .cholesky()
            .unwrap();
        assert!(
            a.generalized_reduction(&short, GeneralizedEigenproblem::AxEqualsLambdaBx)
                .is_err()
        );
    }
    #[test]
    fn complex_all_problem_types_and_mutable_view() {
        let c = Complex64::new;
        let a = PackedHermitian::from_vec(2, vec![c(3., 0.), c(1., -0.5), c(2., 0.)]).unwrap();
        let b = crate::PackedSPD::from_vec(2, vec![c(2., 0.), c(0.25, 0.), c(1.5, 0.)]).unwrap();
        let bf = b.cholesky().unwrap();
        for p in PROBLEMS {
            let expected = a.generalized_eigenvalues(&b, p).unwrap();
            let actual = a
                .generalized_reduction(&bf, p)
                .unwrap()
                .eigenvalues()
                .unwrap();
            for (x, y) in actual.iter().zip(expected) {
                assert!((*x - y).abs() < 1e-10);
            }
        }
        let mut storage = [c(3., 0.), c(1., -0.5), c(2., 0.)];
        let mut view = PackedHermitian::from_slice_mut(2, &mut storage).unwrap();
        view.reduce_generalized_in_place(&bf, GeneralizedEigenproblem::AxEqualsLambdaBx)
            .unwrap();
    }

    #[test]
    fn single_precision_and_owned_reuse() {
        let a = PackedSymmetric::from_vec(2, vec![3f32, 1., 2.]).unwrap();
        let b = crate::PackedSPD::from_vec(2, vec![2f32, 0.25, 1.5]).unwrap();
        let bf = b.cholesky().unwrap();
        let pointer = a.as_slice().as_ptr();
        let reduced = a
            .into_generalized_reduction(&bf, GeneralizedEigenproblem::AxEqualsLambdaBx)
            .unwrap();
        assert_eq!(reduced.as_slice().as_ptr(), pointer);

        let c = Complex32::new;
        let a = PackedHermitian::from_vec(2, vec![c(3., 0.), c(1., -0.5), c(2., 0.)]).unwrap();
        let b = crate::PackedSPD::from_vec(2, vec![c(2., 0.), c(0.25, 0.), c(1.5, 0.)]).unwrap();
        let reduced = a
            .generalized_reduction(
                &b.cholesky().unwrap(),
                GeneralizedEigenproblem::BAxEqualsLambdaX,
            )
            .unwrap();
        assert_eq!(reduced.eigenvalues().unwrap().len(), 2);
    }

    #[test]
    fn upper_lapack_path_and_empty_matrix() {
        let a = PackedSymmetric::from_vec(2, vec![3f64, 1., 2.]).unwrap();
        let b = crate::PackedSPD::from_vec(2, vec![2f64, 0.25, 1.5]).unwrap();
        for problem in PROBLEMS {
            let expected = a.generalized_eigenvalues(&b, problem).unwrap();
            let mut upper_a = vec![3f64, 1., 2.];
            let mut upper_b = vec![2f64, 0.25, 1.5];
            let mut info = 0;
            unsafe { lapack::dpptrf(b'U', 2, &mut upper_b, &mut info) };
            assert_eq!(info, 0);
            unsafe {
                <f64 as GeneralizedPackedReduction>::pgst(
                    &[problem.itype()],
                    b'U',
                    2,
                    &mut upper_a,
                    &upper_b,
                    &mut info,
                )
            };
            assert_eq!(info, 0);
            let actual = PackedSymmetric::from_vec(2, upper_a)
                .unwrap()
                .eigenvalues()
                .unwrap();
            for (x, y) in actual.iter().zip(expected) {
                assert!((*x - y).abs() < 1e-10);
            }
        }

        let empty = PackedSymmetric::from_vec(0, Vec::<f64>::new()).unwrap();
        let empty_factor = crate::PackedSPD::from_vec(0, Vec::<f64>::new())
            .unwrap()
            .cholesky()
            .unwrap();
        assert!(
            empty
                .generalized_reduction(&empty_factor, GeneralizedEigenproblem::AxEqualsLambdaBx,)
                .unwrap()
                .as_slice()
                .is_empty()
        );
    }
}
