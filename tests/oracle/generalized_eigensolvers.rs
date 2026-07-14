use matrixpacked::{
    EigenRange, GeneralizedEigenproblem, PackedHermitian, PackedMatrixError, PackedSPD,
    PackedSymmetric,
};
use nalgebra::{DMatrix, DVector};
use num_complex::{Complex32, Complex64};

use super::{
    compare::{OracleScalar, Tolerance, assert_identity_close, assert_matrix_close},
    convert::{hermitian_to_dmatrix, pack_lower_column_major, symmetric_to_dmatrix},
    generate::GeneratedScalar,
};

const PROBLEMS: [GeneralizedEigenproblem; 3] = [
    GeneralizedEigenproblem::AxEqualsLambdaBx,
    GeneralizedEigenproblem::ABxEqualsLambdaX,
    GeneralizedEigenproblem::BAxEqualsLambdaX,
];

fn tolerance<T: OracleScalar>() -> Tolerance<f64> {
    Tolerance::for_scalar::<T>()
}

fn q_f32(n: usize, seed: u64) -> DMatrix<f32> {
    DMatrix::from_fn(n, n, |r, c| {
        (((r + 2) * (c + 1)) as f64 + seed as f64 * 1e-5).cos() as f32
            + if r == c { 1.75 } else { 0.0 }
    })
    .qr()
    .q()
}

fn q_f64(n: usize, seed: u64) -> DMatrix<f64> {
    DMatrix::from_fn(n, n, |r, c| {
        (((r + 2) * (c + 1)) as f64 + seed as f64 * 1e-5).cos() + if r == c { 1.75 } else { 0.0 }
    })
    .qr()
    .q()
}

fn q_c32(n: usize, seed: u64) -> DMatrix<Complex32> {
    DMatrix::from_fn(n, n, |r, c| {
        let angle = ((r + 2) * (c + 1)) as f64 + seed as f64 * 1e-5;
        Complex32::new(
            angle.cos() as f32 + if r == c { 1.75 } else { 0.0 },
            (angle * 0.6).sin() as f32,
        )
    })
    .qr()
    .q()
}

fn q_c64(n: usize, seed: u64) -> DMatrix<Complex64> {
    DMatrix::from_fn(n, n, |r, c| {
        let angle = ((r + 2) * (c + 1)) as f64 + seed as f64 * 1e-5;
        Complex64::new(
            angle.cos() + if r == c { 1.75 } else { 0.0 },
            (angle * 0.6).sin(),
        )
    })
    .qr()
    .q()
}

fn projector<T: OracleScalar>(vectors: &DMatrix<T>, first: usize, count: usize) -> DMatrix<T> {
    let cluster = vectors.columns(first, count).into_owned();
    &cluster * cluster.adjoint()
}

macro_rules! generalized_tests {
    ($module:ident, $ty:ty, $real:ty, $packed:ident, $to_dense:ident, $q:ident) => {
        mod $module {
            use super::*;

            fn scalar(value: f64) -> $ty {
                <$ty as GeneratedScalar>::from_f64(value)
            }

            fn lower(n: usize) -> DMatrix<$ty> {
                DMatrix::from_fn(n, n, |row, col| {
                    if row < col {
                        <$ty>::default()
                    } else if row == col {
                        scalar(1.2 + row as f64 * 0.15)
                    } else {
                        scalar(0.04 * (1 + row + 2 * col) as f64)
                    }
                })
            }

            fn standard(n: usize, seed: u64, repeated: bool) -> (DMatrix<$ty>, Vec<$real>) {
                let q = $q(n, seed);
                let values: Vec<$real> = (0..n)
                    .map(|index| {
                        if repeated && index < 2 {
                            -1.0
                        } else {
                            -2.5 + index as f64 * 1.25
                        }
                    })
                    .map(|value| value as $real)
                    .collect();
                let diagonal = DMatrix::from_diagonal(&DVector::from_iterator(
                    n,
                    values.iter().map(|&value| scalar(value as f64)),
                ));
                (&q * diagonal * q.adjoint(), values)
            }

            fn problem_matrix(
                standard: &DMatrix<$ty>,
                l: &DMatrix<$ty>,
                problem: GeneralizedEigenproblem,
            ) -> DMatrix<$ty> {
                match problem {
                    GeneralizedEigenproblem::AxEqualsLambdaBx => l * standard * l.adjoint(),
                    GeneralizedEigenproblem::ABxEqualsLambdaX
                    | GeneralizedEigenproblem::BAxEqualsLambdaX => {
                        let lh = l.adjoint();
                        let left = lh.clone().lu().solve(standard).unwrap();
                        lh.lu().solve(&left.adjoint()).unwrap().adjoint()
                    }
                }
            }

            fn nalgebra_transform(
                a: &DMatrix<$ty>,
                l: &DMatrix<$ty>,
                problem: GeneralizedEigenproblem,
            ) -> DMatrix<$ty> {
                match problem {
                    GeneralizedEigenproblem::AxEqualsLambdaBx => {
                        let left = l.clone().lu().solve(a).unwrap();
                        l.clone().lu().solve(&left.adjoint()).unwrap().adjoint()
                    }
                    GeneralizedEigenproblem::ABxEqualsLambdaX
                    | GeneralizedEigenproblem::BAxEqualsLambdaX => l.adjoint() * a * l,
                }
            }

            fn assert_values_close(left: &[$real], right: &[$real]) {
                assert_eq!(left.len(), right.len());
                for (&left, &right) in left.iter().zip(right) {
                    let threshold = tolerance::<$ty>().abs * 48.0
                        + tolerance::<$ty>().rel * (left.abs().max(right.abs()) as f64) * 48.0;
                    assert!(
                        (left as f64 - right as f64).abs() <= threshold,
                        "generalized eigenvalue {left:?} != {right:?}"
                    );
                }
            }

            fn transformed_vectors(
                vectors: &DMatrix<$ty>,
                l: &DMatrix<$ty>,
                problem: GeneralizedEigenproblem,
            ) -> DMatrix<$ty> {
                match problem {
                    GeneralizedEigenproblem::AxEqualsLambdaBx
                    | GeneralizedEigenproblem::ABxEqualsLambdaX => l.adjoint() * vectors,
                    GeneralizedEigenproblem::BAxEqualsLambdaX => {
                        l.clone().lu().solve(vectors).unwrap()
                    }
                }
            }

            fn validate(
                a: &DMatrix<$ty>,
                b: &DMatrix<$ty>,
                l: &DMatrix<$ty>,
                problem: GeneralizedEigenproblem,
                values: &[$real],
                vector_data: &[$ty],
            ) -> DMatrix<$ty> {
                let n = a.nrows();
                let vectors = DMatrix::from_column_slice(n, values.len(), vector_data);
                let scale = tolerance::<$ty>().rel * 128.0;
                for (index, &value) in values.iter().enumerate() {
                    let vector = vectors.column(index).into_owned();
                    let vector_norm = <$ty as OracleScalar>::real_to_f64(vector.norm());
                    let (left, right) = match problem {
                        GeneralizedEigenproblem::AxEqualsLambdaBx => {
                            (a * &vector, (b * &vector) * scalar(value as f64))
                        }
                        GeneralizedEigenproblem::ABxEqualsLambdaX => {
                            (a * (b * &vector), vector * scalar(value as f64))
                        }
                        GeneralizedEigenproblem::BAxEqualsLambdaX => {
                            (b * (a * &vector), vector * scalar(value as f64))
                        }
                    };
                    let a_norm = <$ty as OracleScalar>::real_to_f64(a.norm());
                    let b_norm = <$ty as OracleScalar>::real_to_f64(b.norm());
                    let lambda = <$ty as OracleScalar>::magnitude(scalar(value as f64));
                    let operator_scale = match problem {
                        GeneralizedEigenproblem::AxEqualsLambdaBx => a_norm + lambda * b_norm,
                        GeneralizedEigenproblem::ABxEqualsLambdaX
                        | GeneralizedEigenproblem::BAxEqualsLambdaX => {
                            a_norm * b_norm + lambda
                        }
                    };
                    let residual = <$ty as OracleScalar>::real_to_f64((&left - &right).norm())
                        / (operator_scale * vector_norm).max(f64::EPSILON);
                    assert!(
                        residual <= scale,
                        "problem={problem:?}, index={index}, residual={residual:e}, threshold={scale:e}"
                    );
                }

                let metric_vectors = match problem {
                    GeneralizedEigenproblem::AxEqualsLambdaBx
                    | GeneralizedEigenproblem::ABxEqualsLambdaX => b * &vectors,
                    GeneralizedEigenproblem::BAxEqualsLambdaX => {
                        b.clone().cholesky().unwrap().solve(&vectors)
                    }
                };
                assert_identity_close(
                    &(vectors.adjoint() * metric_vectors),
                    Tolerance {
                        abs: tolerance::<$ty>().abs * 64.0,
                        rel: tolerance::<$ty>().rel * 64.0,
                    },
                );
                let standard_vectors = transformed_vectors(&vectors, l, problem);
                assert_identity_close(
                    &(standard_vectors.adjoint() * &standard_vectors),
                    Tolerance {
                        abs: tolerance::<$ty>().abs * 64.0,
                        rel: tolerance::<$ty>().rel * 64.0,
                    },
                );
                standard_vectors
            }

            #[test]
            fn generalized_drivers_all_problem_types_match_nalgebra_and_invariants() {
                for n in [1, 2, 5, 8] {
                    let l = lower(n);
                    let b = &l * l.adjoint();
                    let b_data = pack_lower_column_major(&b);
                    let (standard, expected) = standard(n, 0x71_0000 + n as u64, false);
                    for problem in PROBLEMS {
                        let a = problem_matrix(&standard, &l, problem);
                        let a_data = pack_lower_column_major(&a);
                        let packed_a = $packed::from_vec(n, a_data.clone()).unwrap();
                        let packed_b = PackedSPD::from_vec(n, b_data.clone()).unwrap();
                        let basic = packed_a
                            .generalized_eigendecomposition(&packed_b, problem)
                            .unwrap();
                        let dc = packed_a
                            .generalized_eigendecomposition_divide_conquer(&packed_b, problem)
                            .unwrap();
                        let selected = packed_a
                            .generalized_selected_eigendecomposition(
                                &packed_b,
                                problem,
                                EigenRange::All,
                            )
                            .unwrap();
                        assert_eq!(packed_a.as_slice(), a_data.as_slice());
                        assert_eq!(packed_b.as_slice(), b_data.as_slice());

                        let transformed = nalgebra_transform(&a, &l, problem);
                        let mut reference: Vec<$real> = transformed
                            .symmetric_eigen()
                            .eigenvalues
                            .iter()
                            .copied()
                            .collect();
                        reference.sort_by(|left, right| left.partial_cmp(right).unwrap());
                        for values in [&basic.eigenvalues, &dc.eigenvalues, &selected.eigenvalues] {
                            assert_values_close(values, &expected);
                            assert_values_close(values, &reference);
                        }
                        assert_values_close(
                            &packed_a
                                .generalized_eigenvalues(&packed_b, problem)
                                .unwrap(),
                            &basic.eigenvalues,
                        );
                        assert_values_close(
                            &packed_a
                                .generalized_eigenvalues_divide_conquer(&packed_b, problem)
                                .unwrap(),
                            &dc.eigenvalues,
                        );
                        assert_values_close(
                            &packed_a
                                .generalized_selected_eigenvalues(
                                    &packed_b,
                                    problem,
                                    EigenRange::All,
                                )
                                .unwrap(),
                            &selected.eigenvalues,
                        );

                        validate(
                            &a,
                            &b,
                            &l,
                            problem,
                            &basic.eigenvalues,
                            basic.eigenvectors.as_ref().unwrap(),
                        );
                        validate(
                            &a,
                            &b,
                            &l,
                            problem,
                            &dc.eigenvalues,
                            dc.eigenvectors.as_ref().unwrap(),
                        );
                        validate(
                            &a,
                            &b,
                            &l,
                            problem,
                            &selected.eigenvalues,
                            selected.eigenvectors.as_ref().unwrap(),
                        );

                        let consumed = $packed::from_vec(n, a_data.clone())
                            .unwrap()
                            .into_generalized_eigendecomposition(
                                PackedSPD::from_vec(n, b_data.clone()).unwrap(),
                                problem,
                            )
                            .unwrap();
                        validate(
                            &a,
                            &b,
                            &l,
                            problem,
                            &consumed.eigenvalues,
                            consumed.eigenvectors.as_ref().unwrap(),
                        );
                        let consumed_dc = $packed::from_vec(n, a_data.clone())
                            .unwrap()
                            .into_generalized_eigendecomposition_divide_conquer(
                                PackedSPD::from_vec(n, b_data.clone()).unwrap(),
                                problem,
                            )
                            .unwrap();
                        validate(
                            &a,
                            &b,
                            &l,
                            problem,
                            &consumed_dc.eigenvalues,
                            consumed_dc.eigenvectors.as_ref().unwrap(),
                        );

                        let mut a_view_data = a_data.clone();
                        let a_view = $packed::from_slice_mut(n, &mut a_view_data).unwrap();
                        let mut b_view_data = b_data.clone();
                        let b_view = PackedSPD::from_slice_mut(n, &mut b_view_data).unwrap();
                        a_view.generalized_eigenvalues(&b_view, problem).unwrap();
                        assert_eq!(a_view_data, a_data);
                        assert_eq!(b_view_data, b_data);
                    }
                }
            }

            #[test]
            fn generalized_selected_ranges_and_repeated_subspaces() {
                let n = 5;
                let l = lower(n);
                let b = &l * l.adjoint();
                let b_data = pack_lower_column_major(&b);
                for problem in PROBLEMS {
                    let (standard_matrix, expected) = standard(n, 0x72_0000, false);
                    let a = problem_matrix(&standard_matrix, &l, problem);
                    let packed_a = $packed::from_vec(n, pack_lower_column_major(&a)).unwrap();
                    let packed_b = PackedSPD::from_vec(n, b_data.clone()).unwrap();
                    assert_values_close(
                        &packed_a
                            .generalized_selected_eigenvalues(
                                &packed_b,
                                problem,
                                EigenRange::Index { first: 1, last: 3 },
                            )
                            .unwrap(),
                        &expected[1..=3],
                    );
                    assert_values_close(
                        &packed_a
                            .generalized_selected_eigenvalues(
                                &packed_b,
                                problem,
                                EigenRange::Value {
                                    lower: (expected[0] + expected[1]) / (2.0 as $real),
                                    upper: (expected[3] + expected[4]) / (2.0 as $real),
                                },
                            )
                            .unwrap(),
                        &expected[1..=3],
                    );
                    let single = packed_a
                        .generalized_selected_eigendecomposition(
                            &packed_b,
                            problem,
                            EigenRange::Index { first: 2, last: 2 },
                        )
                        .unwrap();
                    assert_eq!(single.count, 1);
                    validate(
                        &a,
                        &b,
                        &l,
                        problem,
                        &single.eigenvalues,
                        single.eigenvectors.as_ref().unwrap(),
                    );

                    for invalid in [
                        EigenRange::Index { first: 3, last: 2 },
                        EigenRange::Index { first: 0, last: n },
                    ] {
                        assert!(matches!(
                            packed_a.generalized_selected_eigenvalues(&packed_b, problem, invalid,),
                            Err(PackedMatrixError::InvalidEigenRange { .. })
                        ));
                    }
                    assert!(matches!(
                        packed_a.generalized_selected_eigenvalues(
                            &packed_b,
                            problem,
                            EigenRange::Value {
                                lower: expected[2],
                                upper: expected[2],
                            },
                        ),
                        Err(PackedMatrixError::InvalidEigenRange { .. })
                    ));

                    let (repeated_standard, _) = standard(n, 0x73_0000, true);
                    let repeated_a = problem_matrix(&repeated_standard, &l, problem);
                    let repeated =
                        $packed::from_vec(n, pack_lower_column_major(&repeated_a)).unwrap();
                    let basic = repeated
                        .generalized_eigendecomposition(&packed_b, problem)
                        .unwrap();
                    let dc = repeated
                        .generalized_eigendecomposition_divide_conquer(&packed_b, problem)
                        .unwrap();
                    let basic_standard = transformed_vectors(
                        &DMatrix::from_column_slice(n, n, basic.eigenvectors.as_ref().unwrap()),
                        &l,
                        problem,
                    );
                    let dc_standard = transformed_vectors(
                        &DMatrix::from_column_slice(n, n, dc.eigenvectors.as_ref().unwrap()),
                        &l,
                        problem,
                    );
                    assert_matrix_close(
                        &projector(&basic_standard, 0, 2),
                        &projector(&dc_standard, 0, 2),
                        Tolerance {
                            abs: tolerance::<$ty>().abs * 64.0,
                            rel: tolerance::<$ty>().rel * 64.0,
                        },
                    );
                }
            }

            #[test]
            fn generalized_reduction_matches_nalgebra_and_overwrite_contracts() {
                let n = 6;
                let l = lower(n);
                let b = &l * l.adjoint();
                let b_packed = PackedSPD::from_vec(n, pack_lower_column_major(&b)).unwrap();
                let factor = b_packed.cholesky().unwrap();
                let factor_data = factor.as_slice().to_vec();
                let (standard, _) = standard(n, 0x74_0000, false);
                for problem in PROBLEMS {
                    let a = problem_matrix(&standard, &l, problem);
                    let a_data = pack_lower_column_major(&a);
                    let packed = $packed::from_vec(n, a_data.clone()).unwrap();
                    let reduced = packed.generalized_reduction(&factor, problem).unwrap();
                    assert_eq!(packed.as_slice(), a_data.as_slice());
                    assert_eq!(factor.as_slice(), factor_data.as_slice());
                    assert_matrix_close(
                        &$to_dense(&reduced),
                        &nalgebra_transform(&a, &l, problem),
                        Tolerance {
                            abs: tolerance::<$ty>().abs * 64.0,
                            rel: tolerance::<$ty>().rel * 64.0,
                        },
                    );

                    let mut view_data = a_data.clone();
                    let mut view = $packed::from_slice_mut(n, &mut view_data).unwrap();
                    view.reduce_generalized_in_place(&factor, problem).unwrap();
                    assert_matrix_close(
                        &$to_dense(&view),
                        &nalgebra_transform(&a, &l, problem),
                        Tolerance {
                            abs: tolerance::<$ty>().abs * 64.0,
                            rel: tolerance::<$ty>().rel * 64.0,
                        },
                    );
                    drop(view);
                    assert_ne!(view_data, a_data);

                    let owned = $packed::from_vec(n, a_data).unwrap();
                    let pointer = owned.as_slice().as_ptr();
                    let reduced = owned.into_generalized_reduction(&factor, problem).unwrap();
                    assert_eq!(reduced.as_slice().as_ptr(), pointer);
                }
            }
        }
    };
}

generalized_tests!(
    generalized_f32,
    f32,
    f32,
    PackedSymmetric,
    symmetric_to_dmatrix,
    q_f32
);
generalized_tests!(
    generalized_f64,
    f64,
    f64,
    PackedSymmetric,
    symmetric_to_dmatrix,
    q_f64
);
generalized_tests!(
    generalized_c32,
    Complex32,
    f32,
    PackedHermitian,
    hermitian_to_dmatrix,
    q_c32
);
generalized_tests!(
    generalized_c64,
    Complex64,
    f64,
    PackedHermitian,
    hermitian_to_dmatrix,
    q_c64
);

#[test]
fn generalized_failures_and_empty_problem() {
    let a = PackedSymmetric::from_vec(2, vec![2.0f64, 0.25, 3.0]).unwrap();
    let short_b = PackedSPD::from_vec(1, vec![1.0f64]).unwrap();
    assert_eq!(
        a.generalized_eigenvalues(&short_b, GeneralizedEigenproblem::AxEqualsLambdaBx),
        Err(PackedMatrixError::DimensionMismatch { left: 2, right: 1 })
    );

    let indefinite_b = PackedSPD::from_vec(2, vec![1.0f64, 0.0, -1.0]).unwrap();
    assert!(matches!(
        a.generalized_eigenvalues(&indefinite_b, GeneralizedEigenproblem::AxEqualsLambdaBx,),
        Err(PackedMatrixError::PositiveDefinitenessFailure { .. })
    ));
    assert!(matches!(
        PackedSymmetric::from_vec(3, vec![1.0f64, 2.0]),
        Err(PackedMatrixError::InvalidLength {
            n: 3,
            expected: 6,
            actual: 2,
        })
    ));

    let empty_a = PackedSymmetric::<f64>::from_vec(0, vec![]).unwrap();
    let empty_b = PackedSPD::<f64>::from_vec(0, vec![]).unwrap();
    let result = empty_a
        .generalized_eigendecomposition(&empty_b, GeneralizedEigenproblem::AxEqualsLambdaBx)
        .unwrap();
    assert!(result.eigenvalues.is_empty());
    assert!(result.eigenvectors.unwrap().is_empty());
}
