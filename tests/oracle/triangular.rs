use matrixpacked::{
    ConditionNorm, Diagonal, MatrixNorm, PackedLower, PackedMatrixError, PackedUpper, Transpose,
};
use nalgebra::{DMatrix, DVector};
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use proptest::prelude::*;

use super::{
    compare::{
        OracleScalar, Tolerance, assert_identity_close, assert_matrix_close, assert_slice_close,
        matrix_rhs_residual, vector_residual,
    },
    convert::{
        decode_lower_packed, decode_upper_packed, pack_lower_column_major, pack_upper_column_major,
    },
    generate::{GeneratedScalar, nonsingular_lower, nonsingular_upper, vector},
};

const DIMS: [usize; 6] = [0, 1, 2, 3, 5, 8];
const OPS: [Transpose; 3] = [
    Transpose::None,
    Transpose::Transpose,
    Transpose::ConjugateTranspose,
];
const DIAGONALS: [Diagonal; 2] = [Diagonal::NonUnit, Diagonal::Unit];

fn tolerance<T: OracleScalar>() -> Tolerance<f64> {
    Tolerance::for_scalar::<T>()
}

fn logical<T: OracleScalar>(matrix: &DMatrix<T>, diagonal: Diagonal) -> DMatrix<T> {
    let mut out = matrix.clone();
    if diagonal == Diagonal::Unit {
        for i in 0..out.nrows() {
            out[(i, i)] = T::one();
        }
    }
    out
}

fn operated<T: OracleScalar>(matrix: &DMatrix<T>, op: Transpose, diagonal: Diagonal) -> DMatrix<T> {
    let matrix = logical(matrix, diagonal);
    match op {
        Transpose::None => matrix,
        Transpose::Transpose => matrix.transpose(),
        Transpose::ConjugateTranspose => matrix.adjoint(),
    }
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

fn strided<T: Copy>(logical: &[T], increment: i32, padding: T) -> Vec<T> {
    if logical.is_empty() {
        return Vec::new();
    }
    let stride = increment.unsigned_abs() as usize;
    let mut out = vec![padding; 1 + (logical.len() - 1) * stride];
    for (i, &value) in logical.iter().enumerate() {
        let position = if increment > 0 {
            i * stride
        } else {
            (logical.len() - 1 - i) * stride
        };
        out[position] = value;
    }
    out
}

fn unstride<T: Copy>(physical: &[T], n: usize, increment: i32) -> Vec<T> {
    let stride = increment.unsigned_abs() as usize;
    (0..n)
        .map(|i| {
            let position = if increment > 0 {
                i * stride
            } else {
                (n - 1 - i) * stride
            };
            physical[position]
        })
        .collect()
}

macro_rules! triangular_oracle_tests {
    ($module:ident, $ty:ty, $real:ty, $packed:ident, $generate:ident, $pack:ident, $decode:ident) => {
        mod $module {
            use super::*;

            #[test]
            fn tpmv_and_tpsv_single_rhs_all_modes_and_storage_forms() {
                for n in DIMS {
                    let mut full = $generate::<$ty>(n, 0x41_0000 + n as u64, 1.5);
                    for i in 0..n { full[(i, i)] = <$ty as GeneratedScalar>::from_f64(2.0 + i as f64 / 7.0); }
                    let data = $pack(&full);
                    let x = vector::<$ty>(n, 0x41_1000 + n as u64);
                    let owned = $packed::from_vec(n, data.clone()).unwrap();
                    let view = $packed::from_slice(n, &data).unwrap();
                    for op in OPS {
                        for diagonal in DIAGONALS {
                            let a = operated(&full, op, diagonal);
                            let b = &a * &x;
                            let product = owned.mul_vector_op(x.as_slice(), op, diagonal).unwrap();
                            assert_slice_close(b.as_slice(), &product, tolerance::<$ty>(), n);
                            let solved = owned.solve_vector_blas(b.as_slice(), op, diagonal).unwrap();
                            assert_slice_close(x.as_slice(), &solved, tolerance::<$ty>(), n);
                            assert!(vector_residual(&a, &DVector::from_vec(solved), &b) <= tolerance::<$ty>().rel * 8.0 + f64::EPSILON);

                            let mut through_view = b.as_slice().to_vec();
                            view.solve_vector_blas_in_place(&mut through_view, op, diagonal).unwrap();
                            assert_slice_close(x.as_slice(), &through_view, tolerance::<$ty>(), n);

                            let mut mutable_data = data.clone();
                            let mutable = $packed::from_slice_mut(n, &mut mutable_data).unwrap();
                            let mut through_mut = b.as_slice().to_vec();
                            mutable.solve_vector_blas_in_place(&mut through_mut, op, diagonal).unwrap();
                            assert_slice_close(x.as_slice(), &through_mut, tolerance::<$ty>(), n);
                            assert_eq!(mutable_data, data, "solve must not mutate packed storage");
                        }
                    }
                    let b = &full * &x;
                    assert_slice_close(x.as_slice(), &owned.solve_vector(b.as_slice()).unwrap(), tolerance::<$ty>(), n);
                }
            }

            #[test]
            fn tpsv_signed_strides_preserve_padding() {
                let n = 5;
                let full = $generate::<$ty>(n, 0x41_2000, 1.5);
                let matrix = $packed::from_vec(n, $pack(&full)).unwrap();
                let expected = vector::<$ty>(n, 0x41_2100);
                let b = &full * &expected;
                let padding = <$ty as GeneratedScalar>::from_f64(91.0);
                for increment in [1, 2, 3, -1, -2, -3] {
                    let mut physical = strided(b.as_slice(), increment, padding);
                    let before = physical.clone();
                    matrix.solve_vector_blas_strided_in_place(&mut physical, increment, Transpose::None, Diagonal::NonUnit).unwrap();
                    assert_slice_close(expected.as_slice(), &unstride(&physical, n, increment), tolerance::<$ty>(), n);
                    let stride = increment.unsigned_abs() as usize;
                    for i in 0..physical.len() { if i % stride != 0 { assert_eq!(physical[i], before[i]); } }
                }
            }

            #[test]
            fn tptrs_multiple_rhs_counts_all_modes_and_storage_forms() {
                for n in DIMS {
                    let full = $generate::<$ty>(n, 0x41_3000 + n as u64, 1.5);
                    let data = $pack(&full);
                    for nrhs in [0, 1, 2, 4] {
                        let x = DMatrix::from_fn(n, nrhs, |row, col| <$ty as GeneratedScalar>::from_f64(0.25 + row as f64 / 11.0 - col as f64 / 13.0));
                        for op in OPS {
                            for diagonal in DIAGONALS {
                                let a = operated(&full, op, diagonal);
                                let b = &a * &x;
                                for storage in 0..3 {
                                    let mut actual = b.as_slice().to_vec();
                                    match storage {
                                        0 => $packed::from_vec(n, data.clone()).unwrap().solve_many_in_place(&mut actual, nrhs, op, diagonal).unwrap(),
                                        1 => $packed::from_slice(n, &data).unwrap().solve_many_in_place(&mut actual, nrhs, op, diagonal).unwrap(),
                                        _ => { let mut backing = data.clone(); $packed::from_slice_mut(n, &mut backing).unwrap().solve_many_in_place(&mut actual, nrhs, op, diagonal).unwrap(); },
                                    }
                                    assert_slice_close(x.as_slice(), &actual, tolerance::<$ty>(), n);
                                    let actual = DMatrix::from_column_slice(n, nrhs, &actual);
                                    assert!(matrix_rhs_residual(&a, &actual, &b) <= tolerance::<$ty>().rel * 8.0 + f64::EPSILON);
                                }
                            }
                        }
                    }
                }
            }

            #[test]
            fn tptri_owned_consuming_and_view_mut_are_two_sided_inverses() {
                for n in DIMS {
                    let full = $generate::<$ty>(n, 0x41_4000 + n as u64, 1.5);
                    let data = $pack(&full);
                    for diagonal in DIAGONALS {
                        let a = logical(&full, diagonal);
                        let matrix = $packed::from_vec(n, data.clone()).unwrap();
                        let inverse = matrix.inverse_with_diagonal(diagonal).unwrap();
                        let inv_full = logical(&$decode(n, inverse.as_slice()), diagonal);
                        assert_identity_close(&(&a * &inv_full), tolerance::<$ty>());
                        assert_identity_close(&(&inv_full * &a), tolerance::<$ty>());

                        let consumed = $packed::from_vec(n, data.clone()).unwrap().into_inverse_with_diagonal(diagonal).unwrap();
                        assert_matrix_close(&inv_full, &logical(&$decode(n, consumed.as_slice()), diagonal), tolerance::<$ty>());

                        let viewed = $packed::from_slice(n, &data).unwrap().inverse_with_diagonal(diagonal).unwrap();
                        assert_matrix_close(&inv_full, &logical(&$decode(n, viewed.as_slice()), diagonal), tolerance::<$ty>());

                        let mut backing = data.clone();
                        $packed::from_slice_mut(n, &mut backing).unwrap().inverse_in_place_with_diagonal(diagonal).unwrap();
                        assert_matrix_close(&inv_full, &logical(&$decode(n, &backing), diagonal), tolerance::<$ty>());
                    }
                }
            }

            #[test]
            fn lantp_matches_explicit_norms_and_unit_diagonal() {
                for n in DIMS {
                    let full = $generate::<$ty>(n, 0x41_5000 + n as u64, 1.5);
                    let data = $pack(&full);
                    let matrix = $packed::from_vec(n, data.clone()).unwrap();
                    for diagonal in DIAGONALS {
                        let logical = logical(&full, diagonal);
                        for norm in [MatrixNorm::MaxAbs, MatrixNorm::One, MatrixNorm::Infinity, MatrixNorm::Frobenius] {
                            let expected = explicit_norm(&logical, norm);
                            let actual = matrix.matrix_norm(norm, diagonal).unwrap() as f64;
                            let tol = tolerance::<$ty>();
                            assert!((actual - expected).abs() <= tol.abs * 4.0 + tol.rel * expected.max(actual) * 4.0,
                                "LANTP mismatch n={n}, norm={norm:?}, diagonal={diagonal:?}, expected={expected:e}, actual={actual:e}");
                            let viewed = $packed::from_slice(n, &data).unwrap().matrix_norm(norm, diagonal).unwrap() as f64;
                            let mut mutable_data = data.clone();
                            let mutable = $packed::from_slice_mut(n, &mut mutable_data).unwrap().matrix_norm(norm, diagonal).unwrap() as f64;
                            assert_eq!(actual, viewed);
                            assert_eq!(actual, mutable);
                        }
                    }
                }
            }

            #[test]
            fn tpcon_is_finite_scaled_and_detects_ill_conditioning() {
                let n = 5;
                let identity = DMatrix::<$ty>::identity(n, n);
                let identity_packed = $packed::from_vec(n, $pack(&identity)).unwrap();
                for norm in [ConditionNorm::One, ConditionNorm::Infinity] {
                    let r = identity_packed.rcond(norm, Diagonal::NonUnit).unwrap() as f64;
                    assert!((r - 1.0).abs() <= tolerance::<$ty>().rel * 4.0);
                    assert_eq!(r, identity_packed.reciprocal_condition_number(norm, Diagonal::NonUnit).unwrap() as f64);
                }

                let well = $generate::<$ty>(n, 0x41_6000, 2.0);
                let well_packed = $packed::from_vec(n, $pack(&well)).unwrap();
                let inv = well.clone().try_inverse().unwrap();
                let exact = 1.0 / (explicit_norm(&well, MatrixNorm::One) * explicit_norm(&inv, MatrixNorm::One));
                let estimated = well_packed.rcond(ConditionNorm::One, Diagonal::NonUnit).unwrap() as f64;
                assert!(estimated.is_finite() && estimated >= 0.0 && estimated <= 1.0 + tolerance::<$ty>().rel);
                assert!(estimated >= exact / 100.0 && estimated <= (exact * 100.0).min(1.0 + tolerance::<$ty>().rel), "exact={exact:e}, estimate={estimated:e}");
                let well_data = $pack(&well);
                let viewed = $packed::from_slice(n, &well_data).unwrap().rcond(ConditionNorm::One, Diagonal::NonUnit).unwrap() as f64;
                let mut mutable_data = well_data.clone();
                let mutable = $packed::from_slice_mut(n, &mut mutable_data).unwrap().rcond(ConditionNorm::One, Diagonal::NonUnit).unwrap() as f64;
                assert_eq!(estimated, viewed);
                assert_eq!(estimated, mutable);

                let mut ill = DMatrix::<$ty>::identity(n, n);
                ill[(n - 1, n - 1)] = <$ty as GeneratedScalar>::from_f64(if std::mem::size_of::<$real>() == 4 { 1.0e-4 } else { 1.0e-10 });
                let ill_r = $packed::from_vec(n, $pack(&ill)).unwrap().rcond(ConditionNorm::One, Diagonal::NonUnit).unwrap() as f64;
                assert!(ill_r < estimated * 1.0e-2, "ill={ill_r:e}, well={estimated:e}");

                ill[(n - 1, n - 1)] = <$ty as Zero>::zero();
                let singular_r = $packed::from_vec(n, $pack(&ill)).unwrap().rcond(ConditionNorm::One, Diagonal::NonUnit).unwrap() as f64;
                assert_eq!(singular_r, 0.0);
                let unit_r = $packed::from_vec(n, $pack(&ill)).unwrap().rcond(ConditionNorm::One, Diagonal::Unit).unwrap() as f64;
                assert!(unit_r.is_finite() && unit_r > 0.0);
            }

            #[test]
            fn tprfs_refines_perturbed_solutions_and_reports_each_rhs() {
                let n = 6;
                let full = $generate::<$ty>(n, 0x41_7000, 2.0);
                let matrix = $packed::from_vec(n, $pack(&full)).unwrap();
                for nrhs in [0, 1, 2, 4] {
                    for op in OPS {
                        for diagonal in DIAGONALS {
                            let a = operated(&full, op, diagonal);
                            let exact = DMatrix::from_fn(n, nrhs, |row, col| <$ty as GeneratedScalar>::from_f64(0.4 + row as f64 / 17.0 + col as f64 / 19.0));
                            let b = &a * &exact;
                            let mut perturbed = exact.clone();
                            for (i, value) in perturbed.as_mut_slice().iter_mut().enumerate() {
                                *value += <$ty as GeneratedScalar>::from_f64((i as f64 + 1.0) * if std::mem::size_of::<$real>() == 4 { 2.0e-3 } else { 2.0e-8 });
                            }
                            let before = matrix_rhs_residual(&a, &perturbed, &b);
                            let error_before = <$ty as OracleScalar>::real_to_f64((&perturbed - &exact).norm());
                            let initial = perturbed.clone();
                            let report = matrix.refine_many_in_place(b.as_slice(), perturbed.as_mut_slice(), nrhs, op, diagonal).unwrap();
                            let after = matrix_rhs_residual(&a, &perturbed, &b);
                            let error_after = <$ty as OracleScalar>::real_to_f64((&perturbed - &exact).norm());
                            assert_eq!(report.forward_error.len(), nrhs);
                            assert_eq!(report.backward_error.len(), nrhs);
                            assert!(report.forward_error.iter().all(|&value| { let value = value as f64; value.is_finite() && value >= 0.0 }));
                            assert!(report.backward_error.iter().all(|&value| { let value = value as f64; value.is_finite() && value >= 0.0 }));
                            assert!(after <= before * 1.001 + f64::EPSILON, "refinement worsened residual: before={before:e}, after={after:e}");
                            assert!(error_after <= error_before * 1.001 + f64::EPSILON, "refinement worsened exact-solution error: before={error_before:e}, after={error_after:e}");
                            let data = $pack(&full);
                            let mut through_view = initial.clone();
                            let view_report = $packed::from_slice(n, &data).unwrap().refine_many_in_place(b.as_slice(), through_view.as_mut_slice(), nrhs, op, diagonal).unwrap();
                            assert_eq!(report, view_report);
                            assert_matrix_close(&perturbed, &through_view, tolerance::<$ty>());
                            let mut mutable_data = data.clone();
                            let mut through_mut = initial;
                            let mut_report = $packed::from_slice_mut(n, &mut mutable_data).unwrap().refine_many_in_place(b.as_slice(), through_mut.as_mut_slice(), nrhs, op, diagonal).unwrap();
                            assert_eq!(report, mut_report);
                            assert_matrix_close(&perturbed, &through_mut, tolerance::<$ty>());
                        }
                    }
                }
            }
        }
    };
}

triangular_oracle_tests!(
    lower_f32,
    f32,
    f32,
    PackedLower,
    nonsingular_lower,
    pack_lower_column_major,
    decode_lower_packed
);
triangular_oracle_tests!(
    lower_f64,
    f64,
    f64,
    PackedLower,
    nonsingular_lower,
    pack_lower_column_major,
    decode_lower_packed
);
triangular_oracle_tests!(
    lower_c32,
    Complex32,
    f32,
    PackedLower,
    nonsingular_lower,
    pack_lower_column_major,
    decode_lower_packed
);
triangular_oracle_tests!(
    lower_c64,
    Complex64,
    f64,
    PackedLower,
    nonsingular_lower,
    pack_lower_column_major,
    decode_lower_packed
);
triangular_oracle_tests!(
    upper_f32,
    f32,
    f32,
    PackedUpper,
    nonsingular_upper,
    pack_upper_column_major,
    decode_upper_packed
);
triangular_oracle_tests!(
    upper_f64,
    f64,
    f64,
    PackedUpper,
    nonsingular_upper,
    pack_upper_column_major,
    decode_upper_packed
);
triangular_oracle_tests!(
    upper_c32,
    Complex32,
    f32,
    PackedUpper,
    nonsingular_upper,
    pack_upper_column_major,
    decode_upper_packed
);
triangular_oracle_tests!(
    upper_c64,
    Complex64,
    f64,
    PackedUpper,
    nonsingular_upper,
    pack_upper_column_major,
    decode_upper_packed
);

#[test]
fn triangular_failure_contracts_and_unit_diagonal_semantics() {
    let matrix = PackedLower::from_vec(3, vec![2.0f64, 1.0, -0.5, 3.0, 0.25, 4.0]).unwrap();
    for increment in [0, i32::MIN] {
        assert!(
            matches!(matrix.solve_vector_blas_strided_in_place(&mut [1.0; 3], increment, Transpose::None, Diagonal::NonUnit), Err(PackedMatrixError::InvalidIncrement { increment: value }) if value == increment)
        );
    }
    assert!(matches!(
        matrix.solve_vector_blas_strided_in_place(
            &mut [1.0; 2],
            2,
            Transpose::None,
            Diagonal::NonUnit
        ),
        Err(PackedMatrixError::InvalidVectorLength {
            expected: 5,
            actual: 2
        })
    ));
    assert!(matches!(
        matrix.solve_many_in_place(&mut [1.0; 5], 2, Transpose::None, Diagonal::NonUnit),
        Err(PackedMatrixError::InvalidVectorLength {
            expected: 6,
            actual: 5
        })
    ));
    assert!(matches!(
        matrix.refine_many_in_place(
            &[1.0; 6],
            &mut [1.0; 5],
            2,
            Transpose::None,
            Diagonal::NonUnit
        ),
        Err(PackedMatrixError::InvalidVectorLength {
            expected: 6,
            actual: 5
        })
    ));
    assert!(matches!(
        matrix.solve_many_in_place(&mut [], usize::MAX, Transpose::None, Diagonal::NonUnit),
        Err(PackedMatrixError::DimensionOverflow { n: 3 })
    ));

    let singular = PackedUpper::from_vec(3, vec![1.0f64, 0.5, 0.0, -1.0, 0.25, 2.0]).unwrap();
    assert!(matches!(
        singular.solve_many_in_place(&mut [1.0; 3], 1, Transpose::None, Diagonal::NonUnit),
        Err(PackedMatrixError::FactorizationFailure { index: 2, .. })
    ));
    assert!(matches!(
        singular.inverse(),
        Err(PackedMatrixError::FactorizationFailure { index: 2, .. })
    ));
    let singular_blas = singular
        .solve_vector_blas(&[1.0; 3], Transpose::None, Diagonal::NonUnit)
        .unwrap();
    assert!(singular_blas.iter().any(|value| !value.is_finite()));

    assert!(matches!(
        PackedLower::<f64>::from_vec(3, vec![0.0; 5]),
        Err(PackedMatrixError::InvalidLength {
            expected: 6,
            actual: 5,
            ..
        })
    ));
    assert!(matches!(
        PackedUpper::<f64>::packed_len(usize::MAX),
        Err(PackedMatrixError::DimensionOverflow { .. })
    ));

    let zero_diagonal = PackedLower::from_vec(3, vec![0.0f64, 2.0, -1.0, 0.0, 0.5, 0.0]).unwrap();
    let logical_matrix =
        DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 2.0, 1.0, 0.0, -1.0, 0.5, 1.0]);
    let x = DVector::from_vec(vec![0.5, -1.0, 2.0]);
    let b = &logical_matrix * &x;
    assert_slice_close(
        b.as_slice(),
        &zero_diagonal
            .mul_vector_op(x.as_slice(), Transpose::None, Diagonal::Unit)
            .unwrap(),
        tolerance::<f64>(),
        3,
    );
    assert_slice_close(
        x.as_slice(),
        &zero_diagonal
            .solve_vector_blas(b.as_slice(), Transpose::None, Diagonal::Unit)
            .unwrap(),
        tolerance::<f64>(),
        3,
    );
    let inverse = zero_diagonal.inverse_with_diagonal(Diagonal::Unit).unwrap();
    let inverse = logical(&decode_lower_packed(3, inverse.as_slice()), Diagonal::Unit);
    assert_identity_close(&(logical_matrix * inverse), tolerance::<f64>());
}

proptest! {
    #[test]
    fn randomized_triangular_solve_and_inverse_residuals(n in 1usize..=12, seed in any::<u64>(), upper in any::<bool>()) {
        let x = vector::<f64>(n, seed ^ 0x41_8000);
        if upper {
            let a = nonsingular_upper::<f64>(n, seed, 1.5);
            let packed = PackedUpper::from_vec(n, pack_upper_column_major(&a)).unwrap();
            let b = &a * &x;
            let solved = packed.solve_vector_blas(b.as_slice(), Transpose::None, Diagonal::NonUnit).unwrap();
            prop_assert!(vector_residual(&a, &DVector::from_vec(solved), &b) < 1.0e-11);
            let inv = decode_upper_packed(n, packed.inverse().unwrap().as_slice());
            prop_assert!(matrix_rhs_residual(&a, &inv, &DMatrix::identity(n, n)) < 1.0e-11);
        } else {
            let a = nonsingular_lower::<f64>(n, seed, 1.5);
            let packed = PackedLower::from_vec(n, pack_lower_column_major(&a)).unwrap();
            let b = &a * &x;
            let solved = packed.solve_vector_blas(b.as_slice(), Transpose::None, Diagonal::NonUnit).unwrap();
            prop_assert!(vector_residual(&a, &DVector::from_vec(solved), &b) < 1.0e-11);
            let inv = decode_lower_packed(n, packed.inverse().unwrap().as_slice());
            prop_assert!(matrix_rhs_residual(&a, &inv, &DMatrix::identity(n, n)) < 1.0e-11);
        }
    }
}
