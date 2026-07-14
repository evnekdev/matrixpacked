use matrixpacked::{
    EquilibrationMode, ExpertSolveOptions, LapackScalar, MatrixNorm, PackedHermitian,
    PackedMatrixError, PackedSPD, PackedSymmetric,
};
use nalgebra::{DMatrix, DVector};
use num_complex::{Complex32, Complex64};
use proptest::prelude::*;

use super::{
    compare::{
        OracleScalar, Tolerance, assert_hermitian, assert_identity_close, assert_matrix_close,
        assert_slice_close, assert_symmetric, matrix_rhs_residual, vector_residual,
    },
    convert::{
        decode_lower_packed, hermitian_to_dmatrix, pack_lower_column_major, spd_to_dmatrix,
        symmetric_to_dmatrix,
    },
    generate::{
        GeneratedScalar, column_major_multi_rhs, complex_symmetric, hermitian_indefinite,
        hpd_complex32, hpd_complex64, spd_f32, spd_f64, symmetric_indefinite_f64, vector,
    },
};

const DIMS: [usize; 4] = [1, 2, 4, 7];

fn tolerance<T: OracleScalar>() -> Tolerance<f64> {
    Tolerance::for_scalar::<T>()
}

fn explicit_norm<T: OracleScalar>(matrix: &DMatrix<T>, norm: MatrixNorm) -> f64 {
    match norm {
        MatrixNorm::MaxAbs => matrix.iter().map(|&x| T::magnitude(x)).fold(0.0, f64::max),
        MatrixNorm::One => (0..matrix.ncols())
            .map(|col| {
                (0..matrix.nrows())
                    .map(|row| T::magnitude(matrix[(row, col)]))
                    .sum()
            })
            .fold(0.0, f64::max),
        MatrixNorm::Infinity => (0..matrix.nrows())
            .map(|row| {
                (0..matrix.ncols())
                    .map(|col| T::magnitude(matrix[(row, col)]))
                    .sum()
            })
            .fold(0.0, f64::max),
        MatrixNorm::Frobenius => matrix
            .iter()
            .map(|&x| T::magnitude(x).powi(2))
            .sum::<f64>()
            .sqrt(),
    }
}

fn assert_report<R: Copy + Into<f64>>(forward: &[R], backward: &[R], nrhs: usize) {
    assert_eq!(forward.len(), nrhs);
    assert_eq!(backward.len(), nrhs);
    assert!(forward.iter().all(|&x| {
        let x = x.into();
        x.is_finite() && x >= 0.0
    }));
    assert!(backward.iter().all(|&x| {
        let x = x.into();
        x.is_finite() && x >= 0.0
    }));
}

macro_rules! spd_factorization_tests {
    ($module:ident, $ty:ty, $real:ty, $generate:expr) => {
        mod $module {
            use super::*;

            #[test]
            fn pptrf_reconstructs_and_pptrs_solves_one_and_many_rhs() {
                for n in DIMS {
                    let a: DMatrix<$ty> = ($generate)(n, 0x51_0000 + n as u64);
                    let data = pack_lower_column_major(&a);
                    let matrix = PackedSPD::from_vec(n, data.clone()).unwrap();
                    let factor = matrix.cholesky().unwrap();
                    let l = decode_lower_packed(n, factor.as_slice());
                    assert_matrix_close(
                        &a,
                        &(&l * l.adjoint()),
                        Tolerance {
                            abs: tolerance::<$ty>().abs * 12.0,
                            rel: tolerance::<$ty>().rel * 12.0,
                        },
                    );

                    let x = vector::<$ty>(n, 0x51_1000 + n as u64);
                    let b = &a * &x;
                    let solved = factor.solve_vector(b.as_slice()).unwrap();
                    assert_slice_close(x.as_slice(), &solved, tolerance::<$ty>(), n);
                    assert_slice_close(
                        x.as_slice(),
                        &matrix.solve_vector(b.as_slice()).unwrap(),
                        tolerance::<$ty>(),
                        n,
                    );
                    assert!(
                        vector_residual(&a, &DVector::from_vec(solved), &b)
                            <= tolerance::<$ty>().rel * 8.0
                    );

                    for nrhs in [1, 2, 4] {
                        let expected = DMatrix::from_column_slice(
                            n,
                            nrhs,
                            &column_major_multi_rhs::<$ty>(n, nrhs, 0x51_2000 + nrhs as u64),
                        );
                        let rhs = &a * &expected;
                        let mut actual = rhs.as_slice().to_vec();
                        factor.solve_many_in_place(&mut actual, nrhs).unwrap();
                        assert_slice_close(expected.as_slice(), &actual, tolerance::<$ty>(), n);
                        assert_slice_close(
                            &actual,
                            &matrix.solve_once(rhs.as_slice(), nrhs).unwrap(),
                            tolerance::<$ty>(),
                            n,
                        );
                        assert_slice_close(
                            &actual,
                            &matrix
                                .clone()
                                .into_solve_once(rhs.as_slice().to_vec(), nrhs)
                                .unwrap(),
                            tolerance::<$ty>(),
                            n,
                        );
                        let mut one_shot_storage = data.clone();
                        let mut one_shot_rhs = rhs.as_slice().to_vec();
                        PackedSPD::from_slice_mut(n, &mut one_shot_storage)
                            .unwrap()
                            .solve_once_in_place(&mut one_shot_rhs, nrhs)
                            .unwrap();
                        assert_slice_close(&actual, &one_shot_rhs, tolerance::<$ty>(), n);
                    }

                    let mut backing = data;
                    let view_factor = PackedSPD::from_slice_mut(n, &mut backing)
                        .unwrap()
                        .cholesky_in_place()
                        .unwrap();
                    assert_slice_close(
                        factor.as_slice(),
                        view_factor.as_slice(),
                        tolerance::<$ty>(),
                        n,
                    );
                }
            }

            #[test]
            fn pptri_ppcon_and_pprfs_match_full_oracles() {
                for n in DIMS {
                    let a: DMatrix<$ty> = ($generate)(n, 0x51_3000 + n as u64);
                    let matrix = PackedSPD::from_vec(n, pack_lower_column_major(&a)).unwrap();
                    let factor = matrix.cholesky().unwrap();
                    let anorm = explicit_norm(&a, MatrixNorm::One);
                    let inverse_reference = a.clone().try_inverse().unwrap();
                    let exact_rcond =
                        1.0 / (anorm * explicit_norm(&inverse_reference, MatrixNorm::One));
                    let rcond = factor.rcond(anorm as $real).unwrap() as f64;
                    assert!(
                        rcond.is_finite() && rcond >= 0.0 && rcond <= 1.0 + tolerance::<$ty>().rel
                    );
                    assert!(
                        rcond >= exact_rcond / 100.0 && rcond <= (exact_rcond * 100.0).min(1.01)
                    );

                    let inverse = factor.clone().into_inverse().unwrap();
                    let inverse = spd_to_dmatrix(&inverse);
                    assert_identity_close(
                        &(&a * &inverse),
                        Tolerance {
                            abs: tolerance::<$ty>().abs * 16.0,
                            rel: tolerance::<$ty>().rel * 16.0,
                        },
                    );
                    assert_identity_close(
                        &(&inverse * &a),
                        Tolerance {
                            abs: tolerance::<$ty>().abs * 16.0,
                            rel: tolerance::<$ty>().rel * 16.0,
                        },
                    );
                    assert_hermitian(&inverse, tolerance::<$ty>());

                    let nrhs = 2;
                    let exact = DMatrix::from_column_slice(
                        n,
                        nrhs,
                        &column_major_multi_rhs::<$ty>(n, nrhs, 0x51_4000 + n as u64),
                    );
                    let b = &a * &exact;
                    let mut perturbed = exact.clone();
                    for (i, x) in perturbed.as_mut_slice().iter_mut().enumerate() {
                        *x += <$ty as GeneratedScalar>::from_f64(
                            (i + 1) as f64
                                * if std::mem::size_of::<$real>() == 4 {
                                    1e-3
                                } else {
                                    1e-8
                                },
                        );
                    }
                    let before = matrix_rhs_residual(&a, &perturbed, &b);
                    let report = factor
                        .refine_many_in_place(&matrix, b.as_slice(), perturbed.as_mut_slice(), nrhs)
                        .unwrap();
                    let after = matrix_rhs_residual(&a, &perturbed, &b);
                    assert!(after <= before * 1.001 + f64::EPSILON);
                    assert_report(&report.forward_error, &report.backward_error, nrhs);
                }
            }

            #[test]
            fn ppequ_and_ppsvx_match_explicit_scaling_and_reusable_factor() {
                let n = 5;
                let mut a: DMatrix<$ty> = ($generate)(n, 0x51_5000);
                for i in 0..n {
                    let scale = <$ty as GeneratedScalar>::from_f64(10f64.powi(i as i32 - 2));
                    for j in 0..n {
                        a[(i, j)] *= scale;
                        a[(j, i)] *= scale.conjugate();
                    }
                }
                let data = pack_lower_column_major(&a);
                let matrix = PackedSPD::from_vec(n, data.clone()).unwrap();
                let equil = matrix.equilibration().unwrap();
                assert_eq!(equil.scaling.len(), n);
                assert!(equil.scaling.iter().all(|&x| {
                    let x = x as f64;
                    x.is_finite() && x > 0.0
                }));
                let expected_scaled = DMatrix::from_fn(n, n, |row, col| {
                    a[(row, col)]
                        * <$ty as GeneratedScalar>::from_f64(
                            (equil.scaling[row] * equil.scaling[col]) as f64,
                        )
                });
                let mut scaled = PackedSPD::from_vec(n, data).unwrap();
                scaled.apply_equilibration_in_place(&equil.scaling).unwrap();
                assert_matrix_close(
                    &expected_scaled,
                    &spd_to_dmatrix(&scaled),
                    tolerance::<$ty>(),
                );
                for i in 0..n {
                    assert!(
                        (<$ty as OracleScalar>::magnitude(spd_to_dmatrix(&scaled)[(i, i)]) - 1.0)
                            .abs()
                            <= tolerance::<$ty>().rel * 8.0
                    );
                }

                let nrhs = 2;
                let exact = DMatrix::from_column_slice(
                    n,
                    nrhs,
                    &column_major_multi_rhs::<$ty>(n, nrhs, 0x51_6000),
                );
                let b = &a * &exact;
                let simple = matrix.expert_solve(b.as_slice(), nrhs).unwrap();
                let expert = matrix
                    .expert_solve_with_options(
                        b.as_slice(),
                        nrhs,
                        ExpertSolveOptions {
                            equilibration: EquilibrationMode::Compute,
                        },
                    )
                    .unwrap();
                let reusable = matrix.cholesky().unwrap();
                let mut reusable_solution = b.as_slice().to_vec();
                reusable
                    .solve_many_in_place(&mut reusable_solution, nrhs)
                    .unwrap();
                for result in [&simple, &expert] {
                    assert_slice_close(
                        exact.as_slice(),
                        &result.solution,
                        Tolerance {
                            abs: tolerance::<$ty>().abs * 16.0,
                            rel: tolerance::<$ty>().rel * 16.0,
                        },
                        n,
                    );
                    assert!(result.reciprocal_condition_number as f64 >= 0.0);
                    assert_report(&result.forward_error, &result.backward_error, nrhs);
                }
                assert_slice_close(
                    &reusable_solution,
                    &simple.solution,
                    Tolerance {
                        abs: tolerance::<$ty>().abs * 16.0,
                        rel: tolerance::<$ty>().rel * 16.0,
                    },
                    n,
                );
                let scaling = expert
                    .equilibration
                    .expect("xPPSVX should equilibrate this intentionally unbalanced matrix");
                assert_eq!(scaling.len(), n);
                assert!(scaling.iter().all(|&x| x as f64 > 0.0));
            }
        }
    };
}

spd_factorization_tests!(spd_f32_factorizations, f32, f32, |n, seed| spd_f32(
    n, seed, 1.0
));
spd_factorization_tests!(spd_f64_factorizations, f64, f64, |n, seed| spd_f64(
    n, seed, 1.0
));
spd_factorization_tests!(spd_c32_factorizations, Complex32, f32, |n, seed| {
    hpd_complex32(n, seed, 1.0)
});
spd_factorization_tests!(spd_c64_factorizations, Complex64, f64, |n, seed| {
    hpd_complex64(n, seed, 1.0)
});

macro_rules! indefinite_tests {
    ($module:ident, $ty:ty, $real:ty, $packed:ident, $generate:expr, $expand:ident, $structure:ident) => {
        mod $module {
            use super::*;

            #[test]
            fn factor_solve_inverse_condition_refinement_norm_and_expert_driver() {
                for n in [2, 4, 7] {
                    let a: DMatrix<$ty> = ($generate)(n, 0x52_0000 + n as u64);
                    let matrix = $packed::from_vec(n, pack_lower_column_major(&a)).unwrap();
                    let factor = matrix.factorize().unwrap();
                    assert_eq!(factor.pivots().len(), n);
                    assert!(factor.pivots().iter().all(|pivot| *pivot != 0));
                    let nrhs = if n == 2 { 1 } else { 3 };
                    let exact = DMatrix::from_column_slice(
                        n,
                        nrhs,
                        &column_major_multi_rhs::<$ty>(n, nrhs, 0x52_1000 + n as u64),
                    );
                    let b = &a * &exact;
                    let mut solved = b.as_slice().to_vec();
                    factor.solve_many_in_place(&mut solved, nrhs).unwrap();
                    assert_slice_close(
                        exact.as_slice(),
                        &solved,
                        Tolerance {
                            abs: tolerance::<$ty>().abs * 24.0,
                            rel: tolerance::<$ty>().rel * 24.0,
                        },
                        n,
                    );
                    assert!(
                        matrix_rhs_residual(&a, &DMatrix::from_column_slice(n, nrhs, &solved), &b)
                            <= tolerance::<$ty>().rel * 32.0
                    );
                    let solve_tolerance = Tolerance {
                        abs: tolerance::<$ty>().abs * 24.0,
                        rel: tolerance::<$ty>().rel * 24.0,
                    };
                    assert_slice_close(
                        &solved,
                        &matrix.solve_once(b.as_slice(), nrhs).unwrap(),
                        solve_tolerance,
                        n,
                    );
                    assert_slice_close(
                        &solved,
                        &matrix
                            .clone()
                            .into_solve_once(b.as_slice().to_vec(), nrhs)
                            .unwrap(),
                        solve_tolerance,
                        n,
                    );
                    let mut one_shot_storage = pack_lower_column_major(&a);
                    let mut one_shot_rhs = b.as_slice().to_vec();
                    $packed::from_slice_mut(n, &mut one_shot_storage)
                        .unwrap()
                        .solve_once_in_place(&mut one_shot_rhs, nrhs)
                        .unwrap();
                    assert_slice_close(&solved, &one_shot_rhs, solve_tolerance, n);

                    let inverse = factor.clone().into_inverse().unwrap();
                    let inverse = $expand(&inverse);
                    assert_identity_close(
                        &(&a * &inverse),
                        Tolerance {
                            abs: tolerance::<$ty>().abs * 48.0,
                            rel: tolerance::<$ty>().rel * 48.0,
                        },
                    );
                    assert_identity_close(
                        &(&inverse * &a),
                        Tolerance {
                            abs: tolerance::<$ty>().abs * 48.0,
                            rel: tolerance::<$ty>().rel * 48.0,
                        },
                    );
                    $structure(
                        &inverse,
                        Tolerance {
                            abs: tolerance::<$ty>().abs * 8.0,
                            rel: tolerance::<$ty>().rel * 8.0,
                        },
                    );

                    let anorm = explicit_norm(&a, MatrixNorm::One);
                    let exact_rcond = 1.0 / (anorm * explicit_norm(&inverse, MatrixNorm::One));
                    let rcond = factor.rcond(anorm as $real).unwrap() as f64;
                    assert!(rcond.is_finite() && rcond >= 0.0 && rcond <= 1.01);
                    assert!(
                        rcond >= exact_rcond / 100.0 && rcond <= (exact_rcond * 100.0).min(1.01)
                    );

                    for norm in [
                        MatrixNorm::MaxAbs,
                        MatrixNorm::One,
                        MatrixNorm::Infinity,
                        MatrixNorm::Frobenius,
                    ] {
                        let expected = explicit_norm(&a, norm);
                        let actual = matrix.matrix_norm(norm).unwrap() as f64;
                        assert!(
                            (actual - expected).abs()
                                <= tolerance::<$ty>().abs * 8.0
                                    + tolerance::<$ty>().rel * expected.max(actual) * 8.0
                        );
                    }

                    let mut perturbed = exact.clone();
                    for (i, x) in perturbed.as_mut_slice().iter_mut().enumerate() {
                        *x += <$ty as GeneratedScalar>::from_f64(
                            (i + 1) as f64
                                * if std::mem::size_of::<$real>() == 4 {
                                    1e-3
                                } else {
                                    1e-8
                                },
                        );
                    }
                    let before = matrix_rhs_residual(&a, &perturbed, &b);
                    let report = factor
                        .refine_many_in_place(&matrix, b.as_slice(), perturbed.as_mut_slice(), nrhs)
                        .unwrap();
                    assert!(
                        matrix_rhs_residual(&a, &perturbed, &b) <= before * 1.001 + f64::EPSILON
                    );
                    assert_report(&report.forward_error, &report.backward_error, nrhs);

                    let expert = matrix.expert_solve(b.as_slice(), nrhs).unwrap();
                    assert_slice_close(
                        exact.as_slice(),
                        &expert.solution,
                        Tolerance {
                            abs: tolerance::<$ty>().abs * 32.0,
                            rel: tolerance::<$ty>().rel * 32.0,
                        },
                        n,
                    );
                    assert!(expert.reciprocal_condition_number as f64 >= 0.0);
                    assert_report(&expert.forward_error, &expert.backward_error, nrhs);

                    let mut backing = pack_lower_column_major(&a);
                    let view_factor = $packed::from_slice_mut(n, &mut backing)
                        .unwrap()
                        .factorize_in_place()
                        .unwrap();
                    assert_eq!(factor.pivots(), view_factor.pivots());
                }
            }
        }
    };
}

indefinite_tests!(
    symmetric_f32_factorizations,
    f32,
    f32,
    PackedSymmetric,
    |n, _| symmetric_indefinite_f64(n, 0x52_2000).map(|x| x as f32),
    symmetric_to_dmatrix,
    assert_symmetric
);
indefinite_tests!(
    symmetric_f64_factorizations,
    f64,
    f64,
    PackedSymmetric,
    |n, _| symmetric_indefinite_f64(n, 0x52_3000),
    symmetric_to_dmatrix,
    assert_symmetric
);
indefinite_tests!(
    hermitian_c32_factorizations,
    Complex32,
    f32,
    PackedHermitian,
    |n, _| hermitian_indefinite(n, 0x52_4000).map(|x| Complex32::new(x.re as f32, x.im as f32)),
    hermitian_to_dmatrix,
    assert_hermitian
);
indefinite_tests!(
    hermitian_c64_factorizations,
    Complex64,
    f64,
    PackedHermitian,
    |n, _| hermitian_indefinite(n, 0x52_5000),
    hermitian_to_dmatrix,
    assert_hermitian
);

macro_rules! complex_symmetric_audit {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            for n in [2, 4, 7] {
                let mut a = complex_symmetric::<$ty>(n, 0x53_0000 + n as u64);
                for i in 0..n {
                    a[(i, i)] += <$ty as GeneratedScalar>::from_f64(4.0 + i as f64);
                }
                let matrix = PackedSymmetric::from_vec(n, pack_lower_column_major(&a)).unwrap();
                let x = vector::<$ty>(n, 0x53_1000 + n as u64);
                let b = &a * &x;
                let solved = matrix.solve_vector(b.as_slice()).unwrap();
                assert!(
                    vector_residual(&a, &DVector::from_vec(solved), &b)
                        <= tolerance::<$ty>().rel * 32.0
                );
                let one_shot = matrix.solve_once(b.as_slice(), 1).unwrap();
                assert!(
                    vector_residual(&a, &DVector::from_vec(one_shot), &b)
                        <= tolerance::<$ty>().rel * 32.0
                );
                let inverse = symmetric_to_dmatrix(&matrix.inverse().unwrap());
                assert_identity_close(
                    &(&a * &inverse),
                    Tolerance {
                        abs: tolerance::<$ty>().abs * 32.0,
                        rel: tolerance::<$ty>().rel * 32.0,
                    },
                );
                assert_identity_close(
                    &(&inverse * &a),
                    Tolerance {
                        abs: tolerance::<$ty>().abs * 32.0,
                        rel: tolerance::<$ty>().rel * 32.0,
                    },
                );
                assert_symmetric(&inverse, tolerance::<$ty>());
            }
        }
    };
}
complex_symmetric_audit!(complex_symmetric_c32_uses_direct_residuals, Complex32);
complex_symmetric_audit!(complex_symmetric_c64_uses_direct_residuals, Complex64);

#[test]
fn structured_factorization_failure_contracts() {
    let not_pd = PackedSPD::from_vec(2, vec![1.0f64, 0.0, -1.0]).unwrap();
    assert!(matches!(
        not_pd.cholesky(),
        Err(PackedMatrixError::FactorizationFailure { .. })
    ));
    assert!(matches!(
        not_pd.expert_solve(&[1.0; 2], 1),
        Err(PackedMatrixError::FactorizationFailure { .. })
    ));
    let singular = PackedSymmetric::from_vec(2, vec![1.0f64, 0.0, 0.0]).unwrap();
    assert!(matches!(
        singular.factorize(),
        Err(PackedMatrixError::FactorizationFailure { .. })
    ));
    assert!(matches!(
        singular.expert_solve(&[1.0; 2], 1),
        Err(PackedMatrixError::FactorizationFailure { .. })
    ));

    let good = PackedSPD::from_vec(2, vec![2.0f64, 0.5, 2.0]).unwrap();
    let factor = good.cholesky().unwrap();
    assert!(matches!(
        factor.solve_many_in_place(&mut [1.0; 3], 2),
        Err(PackedMatrixError::InvalidVectorLength {
            expected: 4,
            actual: 3
        })
    ));
    assert!(matches!(
        good.expert_solve(&[1.0; 3], 2),
        Err(PackedMatrixError::InvalidVectorLength {
            expected: 4,
            actual: 3
        })
    ));
    let wrong_dimension = PackedSPD::from_vec(3, vec![2.0f64, 0.0, 0.0, 2.0, 0.0, 2.0]).unwrap();
    assert!(matches!(
        factor.refine_vector_in_place(&wrong_dimension, &[1.0; 2], &mut [1.0; 2]),
        Err(PackedMatrixError::DimensionMismatch { left: 3, right: 2 })
    ));
    assert!(matches!(
        good.clone().apply_equilibration_in_place(&[1.0]),
        Err(PackedMatrixError::InvalidVectorLength {
            expected: 2,
            actual: 1
        })
    ));
    let nonpositive_diagonal = PackedSPD::from_vec(2, vec![1.0f64, 0.0, 0.0]).unwrap();
    assert!(matches!(
        nonpositive_diagonal.equilibration(),
        Err(PackedMatrixError::NonPositiveDiagonal { index: 2 })
    ));
}

#[test]
fn structured_condition_estimates_distinguish_identity_well_and_ill_conditioned() {
    let identity = PackedSPD::from_vec(3, vec![1.0f64, 0.0, 0.0, 1.0, 0.0, 1.0]).unwrap();
    let identity_rcond = identity.cholesky().unwrap().rcond(1.0).unwrap();
    assert!((identity_rcond - 1.0).abs() < 1e-12);

    let well = PackedSPD::from_vec(3, vec![2.0f64, 0.0, 0.0, 3.0, 0.0, 4.0]).unwrap();
    let well_rcond = well.cholesky().unwrap().rcond(4.0).unwrap();
    let ill = PackedSPD::from_vec(3, vec![1.0f64, 0.0, 0.0, 1.0, 0.0, 1e-12]).unwrap();
    let ill_factor = ill.cholesky().unwrap();
    let ill_rcond = ill_factor.rcond(1.0).unwrap();
    assert!(ill_rcond.is_finite() && ill_rcond >= 0.0 && ill_rcond < well_rcond * 1e-6);
    let ill_expert = ill.expert_solve(&[1.0, 1.0, 1e-12], 1).unwrap();
    assert!(ill_expert.reciprocal_condition_number <= well_rcond * 1e-6);

    let symmetric_well =
        PackedSymmetric::from_vec(3, vec![1.0f64, 0.0, 0.0, -2.0, 0.0, 3.0]).unwrap();
    let symmetric_ill =
        PackedSymmetric::from_vec(3, vec![1.0f64, 0.0, 0.0, -2.0, 0.0, 1e-12]).unwrap();
    let symmetric_well_rcond = symmetric_well.factorize().unwrap().rcond(3.0).unwrap();
    let symmetric_ill_rcond = symmetric_ill.factorize().unwrap().rcond(2.0).unwrap();
    assert!(symmetric_ill_rcond < symmetric_well_rcond * 1e-6);

    let c = Complex64::new;
    let hermitian_well = PackedHermitian::from_vec(
        3,
        vec![
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(-2.0, 0.0),
            c(0.0, 0.0),
            c(3.0, 0.0),
        ],
    )
    .unwrap();
    let hermitian_ill = PackedHermitian::from_vec(
        3,
        vec![
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(-2.0, 0.0),
            c(0.0, 0.0),
            c(1e-12, 0.0),
        ],
    )
    .unwrap();
    let hermitian_well_rcond = hermitian_well.factorize().unwrap().rcond(3.0).unwrap();
    let hermitian_ill_rcond = hermitian_ill.factorize().unwrap().rcond(2.0).unwrap();
    assert!(hermitian_ill_rcond < hermitian_well_rcond * 1e-6);
}

proptest! {
    #![proptest_config(super::properties::property_config())]

    #[test]
    fn property_spd_factor_solve_residual(n in 1usize..=12, seed in any::<u64>()) {
        let a = spd_f64(n, seed, 1.0);
        let matrix = PackedSPD::from_vec(n, pack_lower_column_major(&a)).unwrap();
        let x = vector::<f64>(n, seed ^ 0x54_0000);
        let b = &a * &x;
        let solved = matrix.cholesky().unwrap().solve_vector(b.as_slice()).unwrap();
        prop_assert!(vector_residual(&a, &DVector::from_vec(solved), &b) < 1e-11);
    }
}
