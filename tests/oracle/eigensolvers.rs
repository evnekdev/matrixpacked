use matrixpacked::{EigenRange, Eigenvectors, PackedHermitian, PackedMatrixError, PackedSymmetric};
use nalgebra::{DMatrix, DVector};
use num_complex::{Complex32, Complex64};
use num_traits::FromPrimitive;
use proptest::prelude::*;

use super::{
    compare::{OracleScalar, Tolerance, assert_identity_close, assert_matrix_close},
    convert::pack_lower_column_major,
    generate::GeneratedScalar,
};

fn tolerance<T: OracleScalar>() -> Tolerance<f64> {
    Tolerance::for_scalar::<T>()
}

fn q_f32(n: usize, seed: u64) -> DMatrix<f32> {
    DMatrix::from_fn(n, n, |r, c| {
        (((r + 1) * (c + 3)) as f64 + seed as f64 * 1e-5).sin() as f32
            + if r == c { 1.5 } else { 0.0 }
    })
    .qr()
    .q()
}

fn q_f64(n: usize, seed: u64) -> DMatrix<f64> {
    DMatrix::from_fn(n, n, |r, c| {
        (((r + 1) * (c + 3)) as f64 + seed as f64 * 1e-5).sin() + if r == c { 1.5 } else { 0.0 }
    })
    .qr()
    .q()
}

fn q_c32(n: usize, seed: u64) -> DMatrix<Complex32> {
    DMatrix::from_fn(n, n, |r, c| {
        let angle = ((r + 1) * (c + 3)) as f64 + seed as f64 * 1e-5;
        Complex32::new(
            angle.sin() as f32 + if r == c { 1.5 } else { 0.0 },
            (angle * 0.7).cos() as f32,
        )
    })
    .qr()
    .q()
}

fn q_c64(n: usize, seed: u64) -> DMatrix<Complex64> {
    DMatrix::from_fn(n, n, |r, c| {
        let angle = ((r + 1) * (c + 3)) as f64 + seed as f64 * 1e-5;
        Complex64::new(
            angle.sin() + if r == c { 1.5 } else { 0.0 },
            (angle * 0.7).cos(),
        )
    })
    .qr()
    .q()
}

fn ascending(values: &[f64]) -> bool {
    values.windows(2).all(|pair| pair[0] <= pair[1])
}

fn projector<T: OracleScalar>(vectors: &DMatrix<T>, first: usize, count: usize) -> DMatrix<T> {
    let cluster = vectors.columns(first, count).into_owned();
    &cluster * cluster.adjoint()
}

fn assert_phase_aligned<T: OracleScalar>(
    reference: &DVector<T>,
    actual: &DVector<T>,
    tol: Tolerance<f64>,
) {
    let dot = reference.dotc(actual);
    let magnitude = T::magnitude(dot);
    assert!(
        magnitude > 0.9,
        "eigenvectors are not collinear: |dot|={magnitude:e}"
    );
    let phase = dot / T::from_real(T::RealField::from_f64(magnitude).unwrap());
    let aligned = actual * phase.conjugate();
    let error = T::real_to_f64((&aligned - reference).norm());
    assert!(
        error <= tol.abs * 16.0 + tol.rel * 16.0,
        "phase-aligned eigenvector error={error:e}"
    );
}

macro_rules! real_eigensolver_tests {
    ($module:ident, $ty:ty, $q:ident) => {
        mod $module {
            use super::*;

            fn matrix(
                n: usize,
                seed: u64,
                repeated: bool,
            ) -> (DMatrix<$ty>, DMatrix<$ty>, Vec<$ty>) {
                let q = $q(n, seed);
                let values: Vec<$ty> = (0..n)
                    .map(|i| {
                        if repeated && i < 2 {
                            -1.0
                        } else {
                            -2.5 + i as f64 * 1.25
                        }
                    })
                    .map(|x| x as $ty)
                    .collect();
                let d = DMatrix::from_diagonal(&DVector::from_vec(values.clone()));
                let a = &q * d * q.transpose();
                (a, q, values)
            }

            fn validate(a: &DMatrix<$ty>, values: &[$ty], vectors: &[$ty]) {
                let n = a.nrows();
                let count = values.len();
                let v = DMatrix::from_column_slice(n, count, vectors);
                assert_identity_close(
                    &(v.transpose() * &v),
                    Tolerance {
                        abs: tolerance::<$ty>().abs * 24.0,
                        rel: tolerance::<$ty>().rel * 24.0,
                    },
                );
                let scale = <$ty as OracleScalar>::real_to_f64(a.norm()).max(f64::EPSILON);
                for (index, &lambda) in values.iter().enumerate() {
                    let column = v.column(index).into_owned();
                    let residual = a * &column - column * lambda;
                    assert!(
                        <$ty as OracleScalar>::real_to_f64(residual.norm()) / scale
                            <= tolerance::<$ty>().rel * 32.0
                    );
                }
                if count == n {
                    let d = DMatrix::from_diagonal(&DVector::from_column_slice(values));
                    assert_matrix_close(
                        a,
                        &(&v * d * v.transpose()),
                        Tolerance {
                            abs: tolerance::<$ty>().abs * 40.0,
                            rel: tolerance::<$ty>().rel * 40.0,
                        },
                    );
                }
            }

            fn assert_values_close(left: &[$ty], right: &[$ty]) {
                assert_eq!(left.len(), right.len());
                for (&left, &right) in left.iter().zip(right) {
                    let threshold = tolerance::<$ty>().abs * 16.0
                        + tolerance::<$ty>().rel * (left.abs().max(right.abs()) as f64) * 16.0;
                    assert!((left as f64 - right as f64).abs() <= threshold);
                }
            }

            #[test]
            fn spev_spevd_spevx_match_nalgebra_invariants_and_phase_aligned_vectors() {
                for n in [1, 2, 5, 9] {
                    let (a, q, expected) = matrix(n, 0x61_0000 + n as u64, false);
                    let data = pack_lower_column_major(&a);
                    let packed = PackedSymmetric::from_vec(n, data.clone()).unwrap();
                    let nalgebra = a.clone().symmetric_eigen();
                    let mut nalgebra_values: Vec<_> =
                        nalgebra.eigenvalues.iter().copied().collect();
                    nalgebra_values.sort_by(|left, right| left.partial_cmp(right).unwrap());
                    let basic = packed.eigendecomposition().unwrap();
                    let dc = packed.eigendecomposition_divide_conquer().unwrap();
                    let selected = packed.selected_eigendecomposition(EigenRange::All).unwrap();
                    let selected_direct = packed
                        .selected_eigen(EigenRange::All, Eigenvectors::Compute)
                        .unwrap();
                    let selected_with_abstol = packed
                        .selected_eigen_with_abstol(EigenRange::All, Eigenvectors::None, 0.0)
                        .unwrap();
                    assert_eq!(packed.as_slice(), data.as_slice());
                    for values in [&basic.eigenvalues, &dc.eigenvalues, &selected.eigenvalues] {
                        let as_f64: Vec<_> = values.iter().map(|&x| x as f64).collect();
                        assert!(ascending(&as_f64));
                        for ((&actual, &known), &reference) in
                            values.iter().zip(&expected).zip(&nalgebra_values)
                        {
                            let threshold = tolerance::<$ty>().abs * 16.0
                                + tolerance::<$ty>().rel
                                    * (actual.abs().max(known.abs()) as f64)
                                    * 16.0;
                            assert!((actual as f64 - known as f64).abs() <= threshold);
                            assert!((actual as f64 - reference as f64).abs() <= threshold);
                        }
                    }
                    validate(&a, &basic.eigenvalues, basic.eigenvectors.as_ref().unwrap());
                    validate(&a, &dc.eigenvalues, dc.eigenvectors.as_ref().unwrap());
                    let basic_v =
                        DMatrix::from_column_slice(n, n, basic.eigenvectors.as_ref().unwrap());
                    for i in 0..n {
                        assert_phase_aligned(
                            &q.column(i).into_owned(),
                            &basic_v.column(i).into_owned(),
                            tolerance::<$ty>(),
                        );
                    }
                    let selected_v =
                        DMatrix::from_column_slice(n, n, selected.eigenvectors.as_ref().unwrap());
                    assert_matrix_close(
                        &projector(&basic_v, 0, n),
                        &projector(&selected_v, 0, n),
                        tolerance::<$ty>(),
                    );

                    assert_values_close(
                        &packed.eigenvalues().unwrap(),
                        &packed.eigen(Eigenvectors::None).unwrap().eigenvalues,
                    );
                    assert_values_close(
                        &packed.eigenvalues_divide_conquer().unwrap(),
                        &dc.eigenvalues,
                    );
                    assert_values_close(&selected_direct.eigenvalues, &selected.eigenvalues);
                    assert_values_close(&selected_with_abstol.eigenvalues, &selected.eigenvalues);
                    assert_values_close(
                        &PackedSymmetric::from_vec(n, data.clone())
                            .unwrap()
                            .into_eigen(Eigenvectors::None)
                            .unwrap()
                            .eigenvalues,
                        &basic.eigenvalues,
                    );
                    validate(
                        &a,
                        &expected,
                        PackedSymmetric::from_vec(n, data.clone())
                            .unwrap()
                            .into_eigendecomposition()
                            .unwrap()
                            .eigenvectors
                            .unwrap()
                            .as_slice(),
                    );
                    validate(
                        &a,
                        &expected,
                        PackedSymmetric::from_vec(n, data.clone())
                            .unwrap()
                            .into_eigendecomposition_divide_conquer()
                            .unwrap()
                            .eigenvectors
                            .unwrap()
                            .as_slice(),
                    );
                    let view = PackedSymmetric::from_slice(n, &data).unwrap();
                    assert_values_close(&view.eigenvalues().unwrap(), &basic.eigenvalues);
                    let mut mutable_data = data.clone();
                    let mutable = PackedSymmetric::from_slice_mut(n, &mut mutable_data).unwrap();
                    assert_values_close(&mutable.eigenvalues().unwrap(), &basic.eigenvalues);
                    assert_eq!(
                        mutable_data, data,
                        "borrowed ViewMut eigensolvers preserve storage"
                    );
                }
            }

            #[test]
            fn spevx_ranges_boundaries_empty_and_invalid() {
                let values = [
                    -2.0 as $ty,
                    -1.0 as $ty,
                    -1.0 as $ty,
                    0.5 as $ty,
                    3.0 as $ty,
                ];
                let a = DMatrix::from_diagonal(&DVector::from_column_slice(&values));
                let packed = PackedSymmetric::from_vec(5, pack_lower_column_major(&a)).unwrap();
                assert_eq!(
                    packed.selected_eigenvalues(EigenRange::All).unwrap(),
                    values
                );
                assert_eq!(
                    packed
                        .selected_eigenvalues(EigenRange::Index { first: 1, last: 3 })
                        .unwrap(),
                    values[1..=3]
                );
                assert_eq!(
                    packed
                        .selected_eigenvalues(EigenRange::Index { first: 2, last: 2 })
                        .unwrap(),
                    values[2..=2]
                );
                assert_eq!(
                    packed
                        .selected_eigenvalues(EigenRange::Value {
                            lower: -1.0,
                            upper: 0.5
                        })
                        .unwrap(),
                    vec![0.5]
                );
                assert_eq!(
                    packed
                        .selected_eigenvalues(EigenRange::Value {
                            lower: -1.1,
                            upper: -1.0
                        })
                        .unwrap(),
                    vec![-1.0, -1.0]
                );
                assert!(
                    packed
                        .selected_eigenvalues(EigenRange::Value {
                            lower: 4.0,
                            upper: 5.0
                        })
                        .unwrap()
                        .is_empty()
                );
                for range in [
                    EigenRange::Index { first: 3, last: 2 },
                    EigenRange::Index { first: 0, last: 5 },
                ] {
                    assert!(matches!(
                        packed.selected_eigenvalues(range),
                        Err(PackedMatrixError::InvalidEigenRange { .. })
                    ));
                }
                assert!(matches!(
                    packed.selected_eigenvalues(EigenRange::Value {
                        lower: 1.0,
                        upper: 1.0
                    }),
                    Err(PackedMatrixError::InvalidEigenRange { .. })
                ));
                assert!(matches!(
                    packed.selected_eigen_with_abstol(EigenRange::All, Eigenvectors::None, -1.0),
                    Err(PackedMatrixError::InvalidEigenRange { .. })
                ));
                let selected = packed
                    .selected_eigendecomposition(EigenRange::Index { first: 1, last: 3 })
                    .unwrap();
                assert_eq!(selected.count, 3);
                validate(
                    &a,
                    &selected.eigenvalues,
                    selected.eigenvectors.as_ref().unwrap(),
                );
            }

            #[test]
            fn repeated_eigenvalues_compare_cluster_projectors() {
                let n = 6;
                let (a, reference, _) = matrix(n, 0x61_1000, true);
                let packed = PackedSymmetric::from_vec(n, pack_lower_column_major(&a)).unwrap();
                let actual = packed.eigendecomposition().unwrap();
                let vectors =
                    DMatrix::from_column_slice(n, n, actual.eigenvectors.as_ref().unwrap());
                assert_matrix_close(
                    &projector(&reference, 0, 2),
                    &projector(&vectors, 0, 2),
                    Tolerance {
                        abs: tolerance::<$ty>().abs * 32.0,
                        rel: tolerance::<$ty>().rel * 32.0,
                    },
                );
                validate(
                    &a,
                    &actual.eigenvalues,
                    actual.eigenvectors.as_ref().unwrap(),
                );
            }
        }
    };
}

real_eigensolver_tests!(symmetric_f32_eigensolvers, f32, q_f32);
real_eigensolver_tests!(symmetric_f64_eigensolvers, f64, q_f64);

macro_rules! hermitian_eigensolver_tests {
    ($module:ident, $ty:ty, $real:ty, $q:ident) => {
        mod $module {
            use super::*;

            fn matrix(
                n: usize,
                seed: u64,
                repeated: bool,
            ) -> (DMatrix<$ty>, DMatrix<$ty>, Vec<$real>) {
                let q = $q(n, seed);
                let values: Vec<$real> = (0..n)
                    .map(|i| {
                        if repeated && i < 2 {
                            -1.0
                        } else {
                            -2.5 + i as f64 * 1.25
                        }
                    })
                    .map(|x| x as $real)
                    .collect();
                let d = DMatrix::from_diagonal(&DVector::from_iterator(
                    n,
                    values
                        .iter()
                        .map(|&x| <$ty as GeneratedScalar>::from_f64(x as f64)),
                ));
                let a = &q * d * q.adjoint();
                (a, q, values)
            }

            fn validate(a: &DMatrix<$ty>, values: &[$real], vectors: &[$ty]) {
                let n = a.nrows();
                let v = DMatrix::from_column_slice(n, values.len(), vectors);
                assert_identity_close(
                    &(v.adjoint() * &v),
                    Tolerance {
                        abs: tolerance::<$ty>().abs * 24.0,
                        rel: tolerance::<$ty>().rel * 24.0,
                    },
                );
                let scale = <$ty as OracleScalar>::real_to_f64(a.norm()).max(f64::EPSILON);
                for (index, &lambda) in values.iter().enumerate() {
                    let column = v.column(index).into_owned();
                    let residual =
                        a * &column - column * <$ty as GeneratedScalar>::from_f64(lambda as f64);
                    assert!(
                        <$ty as OracleScalar>::real_to_f64(residual.norm()) / scale
                            <= tolerance::<$ty>().rel * 32.0
                    );
                }
                if values.len() == n {
                    let d = DMatrix::from_diagonal(&DVector::from_iterator(
                        n,
                        values
                            .iter()
                            .map(|&x| <$ty as GeneratedScalar>::from_f64(x as f64)),
                    ));
                    assert_matrix_close(
                        a,
                        &(&v * d * v.adjoint()),
                        Tolerance {
                            abs: tolerance::<$ty>().abs * 40.0,
                            rel: tolerance::<$ty>().rel * 40.0,
                        },
                    );
                }
            }

            fn assert_values_close(left: &[$real], right: &[$real]) {
                assert_eq!(left.len(), right.len());
                for (&left, &right) in left.iter().zip(right) {
                    let threshold = tolerance::<$ty>().abs * 16.0
                        + tolerance::<$ty>().rel * (left.abs().max(right.abs()) as f64) * 16.0;
                    assert!((left as f64 - right as f64).abs() <= threshold);
                }
            }

            #[test]
            fn hpev_hpevd_hpevx_agree_with_known_spectrum_and_invariants() {
                for n in [1, 2, 5, 9] {
                    let (a, q, expected) = matrix(n, 0x62_0000 + n as u64, false);
                    let data = pack_lower_column_major(&a);
                    let packed = PackedHermitian::from_vec(n, data.clone()).unwrap();
                    let basic = packed.eigendecomposition().unwrap();
                    let dc = packed.eigendecomposition_divide_conquer().unwrap();
                    let selected = packed.selected_eigendecomposition(EigenRange::All).unwrap();
                    let selected_direct = packed
                        .selected_eigen(EigenRange::All, Eigenvectors::Compute)
                        .unwrap();
                    let selected_with_abstol = packed
                        .selected_eigen_with_abstol(EigenRange::All, Eigenvectors::None, 0.0)
                        .unwrap();
                    assert_eq!(packed.as_slice(), data.as_slice());
                    for values in [&basic.eigenvalues, &dc.eigenvalues, &selected.eigenvalues] {
                        assert!(ascending(
                            &values.iter().map(|&x| x as f64).collect::<Vec<_>>()
                        ));
                        for (&actual, &known) in values.iter().zip(&expected) {
                            let threshold = tolerance::<$ty>().abs * 16.0
                                + tolerance::<$ty>().rel
                                    * (actual.abs().max(known.abs()) as f64)
                                    * 16.0;
                            assert!((actual as f64 - known as f64).abs() <= threshold);
                        }
                    }
                    validate(&a, &basic.eigenvalues, basic.eigenvectors.as_ref().unwrap());
                    validate(&a, &dc.eigenvalues, dc.eigenvectors.as_ref().unwrap());
                    let basic_v =
                        DMatrix::from_column_slice(n, n, basic.eigenvectors.as_ref().unwrap());
                    for i in 0..n {
                        assert_phase_aligned(
                            &q.column(i).into_owned(),
                            &basic_v.column(i).into_owned(),
                            tolerance::<$ty>(),
                        );
                    }
                    let selected_v =
                        DMatrix::from_column_slice(n, n, selected.eigenvectors.as_ref().unwrap());
                    assert_matrix_close(
                        &projector(&basic_v, 0, n),
                        &projector(&selected_v, 0, n),
                        tolerance::<$ty>(),
                    );
                    assert_values_close(
                        &packed.eigenvalues().unwrap(),
                        &packed.eigen(Eigenvectors::None).unwrap().eigenvalues,
                    );
                    assert_values_close(
                        &packed.eigenvalues_divide_conquer().unwrap(),
                        &dc.eigenvalues,
                    );
                    assert_values_close(&selected_direct.eigenvalues, &selected.eigenvalues);
                    assert_values_close(&selected_with_abstol.eigenvalues, &selected.eigenvalues);
                    assert_values_close(
                        &PackedHermitian::from_vec(n, data.clone())
                            .unwrap()
                            .into_eigen(Eigenvectors::None)
                            .unwrap()
                            .eigenvalues,
                        &basic.eigenvalues,
                    );
                    validate(
                        &a,
                        &expected,
                        PackedHermitian::from_vec(n, data.clone())
                            .unwrap()
                            .into_eigendecomposition()
                            .unwrap()
                            .eigenvectors
                            .unwrap()
                            .as_slice(),
                    );
                    validate(
                        &a,
                        &expected,
                        PackedHermitian::from_vec(n, data.clone())
                            .unwrap()
                            .into_eigendecomposition_divide_conquer()
                            .unwrap()
                            .eigenvectors
                            .unwrap()
                            .as_slice(),
                    );
                    let view = PackedHermitian::from_slice(n, &data).unwrap();
                    assert_values_close(&view.eigenvalues().unwrap(), &basic.eigenvalues);
                    let mut mutable_data = data.clone();
                    let mutable = PackedHermitian::from_slice_mut(n, &mut mutable_data).unwrap();
                    assert_values_close(&mutable.eigenvalues().unwrap(), &basic.eigenvalues);
                    assert_eq!(
                        mutable_data, data,
                        "borrowed ViewMut eigensolvers preserve storage"
                    );
                }
            }

            #[test]
            fn hpevx_ranges_repeated_boundaries_empty_and_invalid() {
                let values: [$real; 5] = [-2.0, -1.0, -1.0, 0.5, 3.0];
                let a = DMatrix::from_diagonal(&DVector::from_iterator(
                    5,
                    values
                        .iter()
                        .map(|&x| <$ty as GeneratedScalar>::from_f64(x as f64)),
                ));
                let packed = PackedHermitian::from_vec(5, pack_lower_column_major(&a)).unwrap();
                assert_eq!(
                    packed.selected_eigenvalues(EigenRange::All).unwrap(),
                    values
                );
                assert_eq!(
                    packed
                        .selected_eigenvalues(EigenRange::Index { first: 1, last: 3 })
                        .unwrap(),
                    values[1..=3]
                );
                assert_eq!(
                    packed
                        .selected_eigenvalues(EigenRange::Index { first: 2, last: 2 })
                        .unwrap(),
                    values[2..=2]
                );
                assert_eq!(
                    packed
                        .selected_eigenvalues(EigenRange::Value {
                            lower: -1.0,
                            upper: 0.5
                        })
                        .unwrap(),
                    vec![0.5]
                );
                assert_eq!(
                    packed
                        .selected_eigenvalues(EigenRange::Value {
                            lower: -1.1,
                            upper: -1.0
                        })
                        .unwrap(),
                    vec![-1.0, -1.0]
                );
                assert!(
                    packed
                        .selected_eigenvalues(EigenRange::Value {
                            lower: 4.0,
                            upper: 5.0
                        })
                        .unwrap()
                        .is_empty()
                );
                assert!(matches!(
                    packed.selected_eigenvalues(EigenRange::Index { first: 3, last: 2 }),
                    Err(PackedMatrixError::InvalidEigenRange { .. })
                ));
                assert!(matches!(
                    packed.selected_eigenvalues(EigenRange::Value {
                        lower: 1.0,
                        upper: 1.0
                    }),
                    Err(PackedMatrixError::InvalidEigenRange { .. })
                ));
                assert!(matches!(
                    packed.selected_eigen_with_abstol(EigenRange::All, Eigenvectors::None, -1.0),
                    Err(PackedMatrixError::InvalidEigenRange { .. })
                ));
                let selected = packed
                    .selected_eigendecomposition(EigenRange::Index { first: 1, last: 3 })
                    .unwrap();
                assert_eq!(selected.count, 3);
                validate(
                    &a,
                    &selected.eigenvalues,
                    selected.eigenvectors.as_ref().unwrap(),
                );
            }

            #[test]
            fn repeated_hermitian_eigenvalues_use_cluster_projectors() {
                let n = 6;
                let (a, reference, _) = matrix(n, 0x62_1000, true);
                let packed = PackedHermitian::from_vec(n, pack_lower_column_major(&a)).unwrap();
                let actual = packed.eigendecomposition().unwrap();
                let vectors =
                    DMatrix::from_column_slice(n, n, actual.eigenvectors.as_ref().unwrap());
                assert_matrix_close(
                    &projector(&reference, 0, 2),
                    &projector(&vectors, 0, 2),
                    Tolerance {
                        abs: tolerance::<$ty>().abs * 32.0,
                        rel: tolerance::<$ty>().rel * 32.0,
                    },
                );
                validate(
                    &a,
                    &actual.eigenvalues,
                    actual.eigenvectors.as_ref().unwrap(),
                );
            }
        }
    };
}

hermitian_eigensolver_tests!(hermitian_c32_eigensolvers, Complex32, f32, q_c32);
hermitian_eigensolver_tests!(hermitian_c64_eigensolvers, Complex64, f64, q_c64);

#[test]
fn eigensolver_empty_and_dynamic_range_edges() {
    let empty = PackedSymmetric::<f64>::from_vec(0, vec![]).unwrap();
    assert!(empty.eigenvalues().unwrap().is_empty());
    assert!(
        empty
            .eigendecomposition_divide_conquer()
            .unwrap()
            .eigenvectors
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        empty
            .selected_eigendecomposition(EigenRange::All)
            .unwrap()
            .count,
        0
    );
    assert!(matches!(
        empty.selected_eigenvalues(EigenRange::Index { first: 0, last: 0 }),
        Err(PackedMatrixError::InvalidEigenRange { .. })
    ));

    let dynamic = [-1e12f64, -1e-8, 2.0, 1e8, 1e12];
    let full = DMatrix::from_diagonal(&DVector::from_column_slice(&dynamic));
    let packed = PackedSymmetric::from_vec(5, pack_lower_column_major(&full)).unwrap();
    let actual = packed.eigenvalues_divide_conquer().unwrap();
    for (&a, &e) in actual.iter().zip(&dynamic) {
        assert!((a - e).abs() <= 1e-12 + 1e-12 * a.abs().max(e.abs()));
    }
}

proptest! {
    #[test]
    fn randomized_known_spectrum_residuals(n in 1usize..=16, seed in any::<u64>()) {
        let q = q_f64(n, seed);
        let values: Vec<f64> = (0..n).map(|i| -3.0 + i as f64 * 0.75).collect();
        let a = &q * DMatrix::from_diagonal(&DVector::from_vec(values.clone())) * q.transpose();
        let packed = PackedSymmetric::from_vec(n, pack_lower_column_major(&a)).unwrap();
        let result = packed.eigendecomposition_divide_conquer().unwrap();
        let vectors = DMatrix::from_column_slice(n, n, result.eigenvectors.as_ref().unwrap());
        let scale = a.norm().max(f64::EPSILON);
        for (i, (&actual, &expected)) in result.eigenvalues.iter().zip(&values).enumerate() {
            prop_assert!((actual - expected).abs() <= 1e-10 * actual.abs().max(expected.abs()).max(1.0));
            let v = vectors.column(i).into_owned();
            prop_assert!(((&a * &v - v * actual).norm() / scale) < 1e-10);
        }
    }
}
