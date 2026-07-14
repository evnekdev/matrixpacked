use matrixpacked::{
    LapackScalar, PackedHermitian, PackedLower, PackedMatrixError, PackedSPD, PackedSymmetric,
    PackedUpper,
};
use nalgebra::DMatrix;
use num_complex::{Complex32, Complex64};
use proptest::prelude::*;

use crate::oracle::{
    compare::{OracleScalar, Tolerance, assert_hermitian, assert_matrix_close, assert_symmetric},
    convert::{
        hermitian_to_dmatrix, lower_to_dmatrix, pack_lower_column_major, pack_upper_column_major,
        spd_to_dmatrix, symmetric_to_dmatrix, upper_to_dmatrix,
    },
    generate::{arbitrary_lower, arbitrary_upper, complex_symmetric, hermitian, hpd_complex64},
};

const DIMENSIONS: [usize; 7] = [0, 1, 2, 3, 4, 5, 8];

trait StorageScalar: OracleScalar + LapackScalar + PartialEq {
    fn coordinate(row: usize, col: usize) -> Self;
    fn diagonal(row: usize) -> Self;
    fn spd_entry(row: usize, col: usize) -> Self;
    fn mutation() -> Self;
    fn imaginary_diagonal() -> Self;
}

fn conjugate<T: StorageScalar>(value: T) -> T {
    LapackScalar::conjugate(value)
}

macro_rules! impl_real_storage_scalar {
    ($ty:ty) => {
        impl StorageScalar for $ty {
            fn coordinate(row: usize, col: usize) -> Self {
                (100 * row + col + 1) as $ty
            }
            fn diagonal(row: usize) -> Self {
                (20 + row) as $ty
            }
            fn spd_entry(row: usize, col: usize) -> Self {
                if row == col {
                    Self::diagonal(row)
                } else {
                    (100 * row + col + 1) as $ty * 0.001
                }
            }
            fn mutation() -> Self {
                -37.25
            }
            fn imaginary_diagonal() -> Self {
                41.0
            }
        }
    };
}

impl_real_storage_scalar!(f32);
impl_real_storage_scalar!(f64);

impl StorageScalar for Complex32 {
    fn coordinate(row: usize, col: usize) -> Self {
        Self::new((100 * row + col + 1) as f32, (10 * row) as f32 - col as f32)
    }
    fn diagonal(row: usize) -> Self {
        Self::new((20 + row) as f32, 0.0)
    }
    fn spd_entry(row: usize, col: usize) -> Self {
        if row == col {
            Self::diagonal(row)
        } else {
            Self::coordinate(row, col) * 0.001
        }
    }
    fn mutation() -> Self {
        Self::new(-37.25, 9.5)
    }
    fn imaginary_diagonal() -> Self {
        Self::new(41.0, -7.0)
    }
}

impl StorageScalar for Complex64 {
    fn coordinate(row: usize, col: usize) -> Self {
        Self::new((100 * row + col + 1) as f64, (10 * row) as f64 - col as f64)
    }
    fn diagonal(row: usize) -> Self {
        Self::new((20 + row) as f64, 0.0)
    }
    fn spd_entry(row: usize, col: usize) -> Self {
        if row == col {
            Self::diagonal(row)
        } else {
            Self::coordinate(row, col) * 0.001
        }
    }
    fn mutation() -> Self {
        Self::new(-37.25, 9.5)
    }
    fn imaginary_diagonal() -> Self {
        Self::new(41.0, -7.0)
    }
}

fn lower_full<T: StorageScalar>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, |row, col| {
        if row >= col {
            T::coordinate(row, col)
        } else {
            T::zero()
        }
    })
}

fn upper_full<T: StorageScalar>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, |row, col| {
        if row <= col {
            T::coordinate(row, col)
        } else {
            T::zero()
        }
    })
}

fn symmetric_full<T: StorageScalar>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, |row, col| T::coordinate(row.max(col), row.min(col)))
}

fn spd_full<T: StorageScalar>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, |row, col| {
        let value = T::spd_entry(row.max(col), row.min(col));
        if row >= col { value } else { conjugate(value) }
    })
}

fn hermitian_full<T: StorageScalar>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, |row, col| {
        if row == col {
            T::diagonal(row)
        } else if row > col {
            T::coordinate(row, col)
        } else {
            conjugate(T::coordinate(col, row))
        }
    })
}

fn assert_dimensions(
    n: usize,
    nrows: usize,
    ncols: usize,
    dimension: usize,
    shape: (usize, usize),
) {
    assert_eq!(nrows, n);
    assert_eq!(ncols, n);
    assert_eq!(dimension, n);
    assert_eq!(shape, (n, n));
}

fn exercise_lower<T: StorageScalar>() {
    for n in DIMENSIONS {
        let full = lower_full::<T>(n);
        let expected = pack_lower_column_major(&full);
        let mut owned = PackedLower::from_vec(n, expected.clone()).unwrap();
        assert_dimensions(
            n,
            owned.nrows(),
            owned.ncols(),
            owned.dimension(),
            owned.shape(),
        );
        assert_eq!(owned.as_slice(), expected);
        assert_eq!(
            owned.as_slice().iter().copied().collect::<Vec<_>>(),
            expected
        );
        assert_matrix_close(
            &full,
            &lower_to_dmatrix(&owned),
            Tolerance::for_scalar::<T>(),
        );

        let view = PackedLower::from_slice(n, &expected).unwrap();
        assert_dimensions(
            n,
            view.nrows(),
            view.ncols(),
            view.dimension(),
            view.shape(),
        );
        assert_eq!(view.as_slice(), owned.as_slice());
        assert_matrix_close(
            &full,
            &lower_to_dmatrix(&view),
            Tolerance::for_scalar::<T>(),
        );

        let cloned = owned.clone();
        assert_eq!(cloned.as_slice(), owned.as_slice());
        assert_eq!(cloned.into_vec(), expected);

        assert!(matches!(
            owned.get(n, 0),
            Err(PackedMatrixError::IndexOutOfBounds { .. })
        ));
        assert!(matches!(
            owned.get(0, n),
            Err(PackedMatrixError::IndexOutOfBounds { .. })
        ));
        if n > 1 {
            assert_eq!(owned.get(0, 1).unwrap(), T::zero());
            assert!(matches!(
                owned.try_get(0, 1),
                Err(PackedMatrixError::StructuralZero { .. })
            ));
            assert!(owned.get_stored(0, 1).is_none());

            let value = T::mutation();
            owned.set(n - 1, 0, value).unwrap();
            assert_eq!(owned.get(n - 1, 0).unwrap(), value);
            assert!(matches!(
                owned.set(0, n - 1, value),
                Err(PackedMatrixError::StructuralZero { .. })
            ));

            let mut backing = expected.clone();
            {
                let mut view_mut = PackedLower::from_slice_mut(n, &mut backing).unwrap();
                assert_dimensions(
                    n,
                    view_mut.nrows(),
                    view_mut.ncols(),
                    view_mut.dimension(),
                    view_mut.shape(),
                );
                view_mut.set(n - 1, 0, value).unwrap();
                assert_eq!(view_mut.as_slice(), owned.as_slice());
            }
            assert_eq!(backing, owned.as_slice());
            let mut mutated_full = full.clone();
            mutated_full[(n - 1, 0)] = value;
            assert_matrix_close(
                &mutated_full,
                &lower_to_dmatrix(&owned),
                Tolerance::for_scalar::<T>(),
            );
        }
    }
}

fn exercise_upper<T: StorageScalar>() {
    for n in DIMENSIONS {
        let full = upper_full::<T>(n);
        let expected = pack_upper_column_major(&full);
        let mut owned = PackedUpper::from_vec(n, expected.clone()).unwrap();
        assert_dimensions(
            n,
            owned.nrows(),
            owned.ncols(),
            owned.dimension(),
            owned.shape(),
        );
        assert_eq!(owned.as_slice(), expected);
        assert_eq!(
            owned.as_slice().iter().copied().collect::<Vec<_>>(),
            expected
        );
        assert_matrix_close(
            &full,
            &upper_to_dmatrix(&owned),
            Tolerance::for_scalar::<T>(),
        );

        let view = PackedUpper::from_slice(n, &expected).unwrap();
        assert_dimensions(
            n,
            view.nrows(),
            view.ncols(),
            view.dimension(),
            view.shape(),
        );
        assert_eq!(view.as_slice(), owned.as_slice());
        assert_matrix_close(
            &full,
            &upper_to_dmatrix(&view),
            Tolerance::for_scalar::<T>(),
        );
        assert_eq!(owned.clone().into_vec(), expected);

        assert!(matches!(
            owned.get(n, 0),
            Err(PackedMatrixError::IndexOutOfBounds { .. })
        ));
        if n > 1 {
            assert_eq!(owned.get(1, 0).unwrap(), T::zero());
            assert!(matches!(
                owned.try_get(1, 0),
                Err(PackedMatrixError::StructuralZero { .. })
            ));
            assert!(owned.get_stored(1, 0).is_none());

            let value = T::mutation();
            owned.set(0, n - 1, value).unwrap();
            assert_eq!(owned.get(0, n - 1).unwrap(), value);
            assert!(matches!(
                owned.set(n - 1, 0, value),
                Err(PackedMatrixError::StructuralZero { .. })
            ));

            let mut backing = expected.clone();
            {
                let mut view_mut = PackedUpper::from_slice_mut(n, &mut backing).unwrap();
                assert_dimensions(
                    n,
                    view_mut.nrows(),
                    view_mut.ncols(),
                    view_mut.dimension(),
                    view_mut.shape(),
                );
                view_mut.set(0, n - 1, value).unwrap();
                assert_eq!(view_mut.as_slice(), owned.as_slice());
            }
            assert_eq!(backing, owned.as_slice());
            let mut mutated_full = full.clone();
            mutated_full[(0, n - 1)] = value;
            assert_matrix_close(
                &mutated_full,
                &upper_to_dmatrix(&owned),
                Tolerance::for_scalar::<T>(),
            );
        }
    }
}

fn exercise_symmetric<T: StorageScalar>() {
    for n in DIMENSIONS {
        let full = symmetric_full::<T>(n);
        let expected = pack_lower_column_major(&full);
        let mut owned = PackedSymmetric::from_vec(n, expected.clone()).unwrap();
        assert_dimensions(
            n,
            owned.nrows(),
            owned.ncols(),
            owned.dimension(),
            owned.shape(),
        );
        assert_eq!(owned.as_slice(), expected);
        assert_eq!(
            owned.as_slice().iter().copied().collect::<Vec<_>>(),
            expected
        );
        assert_matrix_close(
            &full,
            &symmetric_to_dmatrix(&owned),
            Tolerance::for_scalar::<T>(),
        );
        assert_symmetric(&symmetric_to_dmatrix(&owned), Tolerance::for_scalar::<T>());

        let view = PackedSymmetric::from_slice(n, &expected).unwrap();
        assert_dimensions(
            n,
            view.nrows(),
            view.ncols(),
            view.dimension(),
            view.shape(),
        );
        assert_eq!(view.as_slice(), owned.as_slice());
        assert_matrix_close(
            &full,
            &symmetric_to_dmatrix(&view),
            Tolerance::for_scalar::<T>(),
        );
        assert_eq!(owned.clone().into_vec(), expected);
        assert!(matches!(
            owned.get(n, 0),
            Err(PackedMatrixError::IndexOutOfBounds { .. })
        ));

        if n > 1 {
            assert_eq!(owned.get(0, n - 1).unwrap(), owned.get(n - 1, 0).unwrap());
            assert_eq!(
                owned.try_get(0, n - 1).unwrap(),
                owned.get_stored(n - 1, 0).unwrap()
            );
            assert!(owned.get_stored(0, n - 1).is_none());

            let value = T::mutation();
            owned.set(0, n - 1, value).unwrap();
            assert_eq!(owned.get(0, n - 1).unwrap(), value);
            assert_eq!(owned.get(n - 1, 0).unwrap(), value);

            let mut backing = expected.clone();
            {
                let mut view_mut = PackedSymmetric::from_slice_mut(n, &mut backing).unwrap();
                assert_dimensions(
                    n,
                    view_mut.nrows(),
                    view_mut.ncols(),
                    view_mut.dimension(),
                    view_mut.shape(),
                );
                view_mut.set(0, n - 1, value).unwrap();
                assert_matrix_close(
                    &symmetric_to_dmatrix(&owned),
                    &symmetric_to_dmatrix(&view_mut),
                    Tolerance::for_scalar::<T>(),
                );
            }
            assert_eq!(backing, owned.as_slice());
        }
    }
}

fn exercise_spd<T: StorageScalar>() {
    for n in DIMENSIONS {
        let full = spd_full::<T>(n);
        assert!(full.clone().cholesky().is_some());
        let expected = pack_lower_column_major(&full);
        let mut owned = PackedSPD::from_vec(n, expected.clone()).unwrap();
        assert_dimensions(
            n,
            owned.nrows(),
            owned.ncols(),
            owned.dimension(),
            owned.shape(),
        );
        assert_eq!(owned.as_slice(), expected);
        assert_eq!(
            owned.as_slice().iter().copied().collect::<Vec<_>>(),
            expected
        );
        assert_matrix_close(&full, &spd_to_dmatrix(&owned), Tolerance::for_scalar::<T>());
        assert_hermitian(&spd_to_dmatrix(&owned), Tolerance::for_scalar::<T>());

        let view = PackedSPD::from_slice(n, &expected).unwrap();
        assert_dimensions(
            n,
            view.nrows(),
            view.ncols(),
            view.dimension(),
            view.shape(),
        );
        assert_eq!(view.as_slice(), owned.as_slice());
        assert_matrix_close(&full, &spd_to_dmatrix(&view), Tolerance::for_scalar::<T>());
        assert_eq!(owned.clone().into_vec(), expected);
        assert!(matches!(
            owned.get(n, 0),
            Err(PackedMatrixError::IndexOutOfBounds { .. })
        ));

        if n > 1 {
            assert_eq!(
                owned.get(0, n - 1).unwrap(),
                conjugate(owned.get(n - 1, 0).unwrap())
            );
            assert!(matches!(
                owned.try_get(0, n - 1),
                Err(PackedMatrixError::StructuralZero { .. })
            ));

            let value = T::mutation();
            owned.set(0, n - 1, value).unwrap();
            assert_eq!(owned.get(0, n - 1).unwrap(), value);
            assert_eq!(owned.get(n - 1, 0).unwrap(), conjugate(value));

            let mut backing = expected.clone();
            {
                let mut view_mut = PackedSPD::from_slice_mut(n, &mut backing).unwrap();
                assert_dimensions(
                    n,
                    view_mut.nrows(),
                    view_mut.ncols(),
                    view_mut.dimension(),
                    view_mut.shape(),
                );
                view_mut.set(0, n - 1, value).unwrap();
                assert_matrix_close(
                    &spd_to_dmatrix(&owned),
                    &spd_to_dmatrix(&view_mut),
                    Tolerance::for_scalar::<T>(),
                );
            }
            assert_eq!(backing, owned.as_slice());
        }
    }
}

fn exercise_hermitian<T: StorageScalar>() {
    for n in DIMENSIONS {
        let full = hermitian_full::<T>(n);
        let expected = pack_lower_column_major(&full);
        let mut owned = PackedHermitian::from_vec(n, expected.clone()).unwrap();
        assert_dimensions(
            n,
            owned.nrows(),
            owned.ncols(),
            owned.dimension(),
            owned.shape(),
        );
        assert_eq!(owned.as_slice(), expected);
        assert_eq!(
            owned.as_slice().iter().copied().collect::<Vec<_>>(),
            expected
        );
        assert_matrix_close(
            &full,
            &hermitian_to_dmatrix(&owned),
            Tolerance::for_scalar::<T>(),
        );
        assert_hermitian(&hermitian_to_dmatrix(&owned), Tolerance::for_scalar::<T>());

        let view = PackedHermitian::from_slice(n, &expected).unwrap();
        assert_dimensions(
            n,
            view.nrows(),
            view.ncols(),
            view.dimension(),
            view.shape(),
        );
        assert_eq!(view.as_slice(), owned.as_slice());
        assert_matrix_close(
            &full,
            &hermitian_to_dmatrix(&view),
            Tolerance::for_scalar::<T>(),
        );
        assert_eq!(owned.clone().into_vec(), expected);
        assert!(matches!(
            owned.get(n, 0),
            Err(PackedMatrixError::IndexOutOfBounds { .. })
        ));

        if n > 1 {
            assert_eq!(
                owned.get(0, n - 1).unwrap(),
                conjugate(owned.get(n - 1, 0).unwrap())
            );
            assert!(matches!(
                owned.try_get(0, n - 1),
                Err(PackedMatrixError::StructuralZero { .. })
            ));

            let value = T::mutation();
            owned.set(0, n - 1, value).unwrap();
            assert_eq!(owned.get(0, n - 1).unwrap(), value);
            assert_eq!(owned.get(n - 1, 0).unwrap(), conjugate(value));

            let mut backing = expected.clone();
            {
                let mut view_mut = PackedHermitian::from_slice_mut(n, &mut backing).unwrap();
                assert_dimensions(
                    n,
                    view_mut.nrows(),
                    view_mut.ncols(),
                    view_mut.dimension(),
                    view_mut.shape(),
                );
                view_mut.set(0, n - 1, value).unwrap();
                assert_matrix_close(
                    &hermitian_to_dmatrix(&owned),
                    &hermitian_to_dmatrix(&view_mut),
                    Tolerance::for_scalar::<T>(),
                );
            }
            assert_eq!(backing, owned.as_slice());
        }
    }
}

macro_rules! scalar_storage_tests {
    ($name:ident, $exercise:ident) => {
        mod $name {
            use super::*;
            #[test]
            fn storage_f32_owned_view_view_mut() {
                $exercise::<f32>();
            }
            #[test]
            fn storage_f64_owned_view_view_mut() {
                $exercise::<f64>();
            }
            #[test]
            fn storage_complex32_owned_view_view_mut() {
                $exercise::<Complex32>();
            }
            #[test]
            fn storage_complex64_owned_view_view_mut() {
                $exercise::<Complex64>();
            }
        }
    };
}

scalar_storage_tests!(lower, exercise_lower);
scalar_storage_tests!(upper, exercise_upper);
scalar_storage_tests!(symmetric, exercise_symmetric);
scalar_storage_tests!(spd, exercise_spd);
scalar_storage_tests!(hermitian, exercise_hermitian);

#[test]
fn storage_packed_length_formula_and_overflow() {
    for n in DIMENSIONS {
        let expected = n * (n + 1) / 2;
        assert_eq!(PackedLower::<f64>::packed_len(n).unwrap(), expected);
        assert_eq!(PackedUpper::<f64>::packed_len(n).unwrap(), expected);
        assert_eq!(PackedSymmetric::<f64>::packed_len(n).unwrap(), expected);
        assert_eq!(PackedSPD::<f64>::packed_len(n).unwrap(), expected);
        assert_eq!(
            PackedHermitian::<Complex64>::packed_len(n).unwrap(),
            expected
        );
    }

    for n in [usize::MAX, usize::MAX - 1, usize::MAX / 2 + 1] {
        assert!(
            matches!(PackedLower::<f64>::packed_len(n), Err(PackedMatrixError::DimensionOverflow { n: value }) if value == n)
        );
        assert!(matches!(
            PackedUpper::<f64>::packed_len(n),
            Err(PackedMatrixError::DimensionOverflow { .. })
        ));
        assert!(matches!(
            PackedSymmetric::<f64>::packed_len(n),
            Err(PackedMatrixError::DimensionOverflow { .. })
        ));
        assert!(matches!(
            PackedSPD::<f64>::packed_len(n),
            Err(PackedMatrixError::DimensionOverflow { .. })
        ));
        assert!(matches!(
            PackedHermitian::<Complex64>::packed_len(n),
            Err(PackedMatrixError::DimensionOverflow { .. })
        ));
    }
}

#[test]
fn storage_invalid_lengths_are_rejected_without_allocation() {
    assert!(matches!(
        PackedLower::<f64>::from_vec(3, vec![0.0; 5]),
        Err(PackedMatrixError::InvalidLength {
            expected: 6,
            actual: 5,
            ..
        })
    ));
    assert!(matches!(
        PackedUpper::<f64, &[f64]>::from_slice(3, &[0.0; 7]),
        Err(PackedMatrixError::InvalidLength {
            expected: 6,
            actual: 7,
            ..
        })
    ));
    assert!(matches!(
        PackedSymmetric::<f64>::from_vec(2, vec![0.0; 2]),
        Err(PackedMatrixError::InvalidLength { .. })
    ));
    assert!(matches!(
        PackedSPD::<f64>::from_vec(1, Vec::new()),
        Err(PackedMatrixError::InvalidLength { .. })
    ));
    assert!(matches!(
        PackedHermitian::<Complex64>::from_vec(0, vec![Complex64::new(0.0, 0.0)]),
        Err(PackedMatrixError::InvalidLength { .. })
    ));
}

#[test]
fn storage_spd_structural_conversions_preserve_exact_storage() {
    let real_data = pack_lower_column_major(&spd_full::<f64>(4));
    let symmetric = PackedSPD::from_vec(4, real_data.clone())
        .unwrap()
        .into_symmetric();
    assert_eq!(symmetric.as_slice(), real_data);
    assert_matrix_close(
        &symmetric_full_from_storage(&real_data, 4, false),
        &symmetric_to_dmatrix(&symmetric),
        Tolerance::for_scalar::<f64>(),
    );

    let complex_data = pack_lower_column_major(&spd_full::<Complex64>(4));
    let hermitian = PackedSPD::from_vec(4, complex_data.clone())
        .unwrap()
        .into_hermitian();
    assert_eq!(hermitian.as_slice(), complex_data);
    assert_matrix_close(
        &symmetric_full_from_storage(&complex_data, 4, true),
        &hermitian_to_dmatrix(&hermitian),
        Tolerance::for_scalar::<Complex64>(),
    );
}

fn symmetric_full_from_storage<T: StorageScalar>(
    data: &[T],
    n: usize,
    conjugate_upper: bool,
) -> DMatrix<T> {
    DMatrix::from_fn(n, n, |row, col| {
        let stored_row = row.max(col);
        let stored_col = row.min(col);
        let index = stored_col * (2 * n - stored_col + 1) / 2 + stored_row - stored_col;
        let value = data[index];
        if conjugate_upper && row < col {
            conjugate(value)
        } else {
            value
        }
    })
}

#[test]
fn storage_hermitian_nonreal_diagonal_write_is_currently_preserved() {
    let mut matrix = PackedHermitian::<Complex64>::identity(2).unwrap();
    let value = <Complex64 as StorageScalar>::imaginary_diagonal();
    matrix.set(1, 1, value).unwrap();

    assert_eq!(matrix.get(1, 1).unwrap(), value);
    assert_eq!(matrix.get_stored(1, 1), Some(&value));
    assert_ne!(matrix.get(1, 1).unwrap(), matrix.get(1, 1).unwrap().conj());
}

#[test]
fn storage_debug_and_display_reflect_logical_values() {
    let lower = PackedLower::from_vec(2, pack_lower_column_major(&lower_full::<f64>(2))).unwrap();
    let upper = PackedUpper::from_vec(2, pack_upper_column_major(&upper_full::<f64>(2))).unwrap();
    let symmetric =
        PackedSymmetric::from_vec(2, pack_lower_column_major(&symmetric_full::<Complex64>(2)))
            .unwrap();
    let hermitian =
        PackedHermitian::from_vec(2, pack_lower_column_major(&hermitian_full::<Complex64>(2)))
            .unwrap();
    let spd = PackedSPD::from_vec(2, pack_lower_column_major(&spd_full::<f64>(2))).unwrap();

    let lower_debug = format!("{lower:?}");
    assert!(lower_debug.contains("0.0"));
    assert!(lower_debug.contains(&format!("{:?}", <f64 as StorageScalar>::coordinate(1, 0))));

    let upper_display = format!("{upper}");
    assert!(upper_display.contains('0'));
    assert!(upper_display.contains(&format!("{}", <f64 as StorageScalar>::coordinate(0, 1))));

    let symmetric_display = format!("{symmetric}");
    let symmetric_value = <Complex64 as StorageScalar>::coordinate(1, 0);
    assert!(
        symmetric_display
            .matches(&format!("{symmetric_value}"))
            .count()
            >= 2
    );

    let hermitian_debug = format!("{hermitian:?}");
    let lower_value = <Complex64 as StorageScalar>::coordinate(1, 0);
    assert!(hermitian_debug.contains(&format!("{lower_value:?}")));
    assert!(hermitian_debug.contains(&format!("{:?}", lower_value.conj())));

    let spd_display = format!("{spd}");
    assert!(spd_display.contains(&format!("{}", <f64 as StorageScalar>::diagonal(0))));
}

proptest! {
    #![proptest_config(super::properties::property_config())]

    #[test]
    fn storage_property_pack_expand_and_views_agree(n in 0usize..12, seed in any::<u64>()) {
        let lower_full = arbitrary_lower::<f64>(n, seed);
        let lower_data = pack_lower_column_major(&lower_full);
        let lower_owned = PackedLower::from_vec(n, lower_data.clone()).unwrap();
        let lower_view = PackedLower::from_slice(n, &lower_data).unwrap();
        assert_matrix_close(&lower_full, &lower_to_dmatrix(&lower_owned), Tolerance::for_scalar::<f64>());
        assert_matrix_close(&lower_to_dmatrix(&lower_owned), &lower_to_dmatrix(&lower_view), Tolerance::for_scalar::<f64>());

        let upper_full = arbitrary_upper::<Complex32>(n, seed ^ 1);
        let upper_data = pack_upper_column_major(&upper_full);
        let upper_owned = PackedUpper::from_vec(n, upper_data.clone()).unwrap();
        let upper_view = PackedUpper::from_slice(n, &upper_data).unwrap();
        assert_matrix_close(&upper_full, &upper_to_dmatrix(&upper_owned), Tolerance::for_scalar::<Complex32>());
        assert_matrix_close(&upper_to_dmatrix(&upper_owned), &upper_to_dmatrix(&upper_view), Tolerance::for_scalar::<Complex32>());

        let symmetric_full = complex_symmetric::<Complex64>(n, seed ^ 2);
        let symmetric_data = pack_lower_column_major(&symmetric_full);
        let symmetric = PackedSymmetric::from_vec(n, symmetric_data).unwrap();
        assert_matrix_close(&symmetric_full, &symmetric_to_dmatrix(&symmetric), Tolerance::for_scalar::<Complex64>());
        assert_symmetric(&symmetric_to_dmatrix(&symmetric), Tolerance::for_scalar::<Complex64>());

        let hermitian_full = hermitian::<Complex64>(n, seed ^ 3);
        let hermitian_data = pack_lower_column_major(&hermitian_full);
        let hermitian = PackedHermitian::from_vec(n, hermitian_data).unwrap();
        assert_matrix_close(&hermitian_full, &hermitian_to_dmatrix(&hermitian), Tolerance::for_scalar::<Complex64>());
        assert_hermitian(&hermitian_to_dmatrix(&hermitian), Tolerance::for_scalar::<Complex64>());

        let hpd_full = hpd_complex64(n, seed ^ 4, 0.5);
        let hpd_data = pack_lower_column_major(&hpd_full);
        let hpd = PackedSPD::from_vec(n, hpd_data).unwrap();
        assert_matrix_close(&hpd_full, &spd_to_dmatrix(&hpd), Tolerance::for_scalar::<Complex64>());
        assert_hermitian(&spd_to_dmatrix(&hpd), Tolerance::for_scalar::<Complex64>());
    }

    #[test]
    fn storage_property_one_symmetric_mutation_changes_only_mirrored_pair(
        n in 2usize..12,
        seed in any::<u64>(),
        row in 0usize..12,
        col in 0usize..12,
    ) {
        let row = row % n;
        let col = col % n;
        let full = complex_symmetric::<Complex64>(n, seed);
        let mut packed = PackedSymmetric::from_vec(n, pack_lower_column_major(&full)).unwrap();
        let before = symmetric_to_dmatrix(&packed);
        let value = Complex64::new(17.0, -3.0);
        packed.set(row, col, value).unwrap();
        let after = symmetric_to_dmatrix(&packed);

        for i in 0..n {
            for j in 0..n {
                let affected = (i == row && j == col) || (i == col && j == row);
                if affected { prop_assert_eq!(after[(i, j)], value); }
                else { prop_assert_eq!(after[(i, j)], before[(i, j)]); }
            }
        }
        assert_symmetric(&after, Tolerance::for_scalar::<Complex64>());
    }

    #[test]
    fn storage_property_hermitian_mutation_preserves_hermitian(n in 1usize..12, seed in any::<u64>()) {
        let full = hermitian::<Complex64>(n, seed);
        let mut packed = PackedHermitian::from_vec(n, pack_lower_column_major(&full)).unwrap();
        let row = if n == 1 { 0 } else { n - 1 };
        let col = 0;
        let value = if row == col { Complex64::new(13.0, 0.0) } else { Complex64::new(13.0, -2.5) };
        packed.set(row, col, value).unwrap();
        assert_hermitian(&hermitian_to_dmatrix(&packed), Tolerance::for_scalar::<Complex64>());
    }
}
