#![cfg(feature = "nalgebra-interop")]

use matrixpacked::{
    ConversionTolerance, FullTriangular, PackedHermitian, PackedHermitianView,
    PackedHermitianViewMut, PackedLower, PackedLowerView, PackedLowerViewMut, PackedSPD,
    PackedSPDView, PackedSPDViewMut, PackedSymmetric, PackedSymmetricView, PackedSymmetricViewMut,
    PackedUpper, PackedUpperView, PackedUpperViewMut, Triangle,
};
use nalgebra::{DMatrix, Scalar};
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use proptest::prelude::*;

use super::{
    convert::{pack_lower_column_major, pack_upper_column_major},
    generate::{hpd_complex64, spd_f64},
    properties::property_config,
};

const DIMENSIONS: [usize; 8] = [0, 1, 2, 3, 4, 5, 8, 9];

trait Coordinate: Scalar + Copy + Zero + PartialEq + core::fmt::Debug {
    fn coordinate(row: usize, column: usize) -> Self;
    fn conjugate(self) -> Self;
    fn real_diagonal(self) -> Self;
    fn positive_diagonal(n: usize) -> Self;
    fn small_phase(row: usize, column: usize) -> Self;
}

macro_rules! impl_real_coordinate {
    ($type:ty) => {
        impl Coordinate for $type {
            fn coordinate(row: usize, column: usize) -> Self {
                (100 * column + row + 1) as Self
            }
            fn conjugate(self) -> Self {
                self
            }
            fn real_diagonal(self) -> Self {
                self
            }
            fn positive_diagonal(n: usize) -> Self {
                (4 * n.max(1) + 1) as Self
            }
            fn small_phase(row: usize, column: usize) -> Self {
                ((row + column + 1) as Self) / 64.0
            }
        }
    };
}
impl_real_coordinate!(f32);
impl_real_coordinate!(f64);

macro_rules! impl_complex_coordinate {
    ($type:ty, $real:ty) => {
        impl Coordinate for $type {
            fn coordinate(row: usize, column: usize) -> Self {
                Self::new(
                    (100 * column + row + 1) as $real,
                    (10 * row + column + 1) as $real,
                )
            }
            fn conjugate(self) -> Self {
                self.conj()
            }
            fn real_diagonal(self) -> Self {
                Self::new(self.re, 0.0)
            }
            fn positive_diagonal(n: usize) -> Self {
                Self::new((4 * n.max(1) + 1) as $real, 0.0)
            }
            fn small_phase(row: usize, column: usize) -> Self {
                Self::new(
                    ((row + column + 1) as $real) / 64.0,
                    ((2 * row + column + 1) as $real) / 128.0,
                )
            }
        }
    };
}
impl_complex_coordinate!(Complex32, f32);
impl_complex_coordinate!(Complex64, f64);

fn source<T: Coordinate>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, T::coordinate)
}

fn triangular<T: Coordinate>(n: usize, triangle: Triangle) -> DMatrix<T> {
    DMatrix::from_fn(n, n, |row, column| {
        let selected = match triangle {
            Triangle::Lower => row >= column,
            Triangle::Upper => row <= column,
        };
        if selected {
            T::coordinate(row, column)
        } else {
            T::zero()
        }
    })
}

fn symmetric<T: Coordinate>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, |row, column| {
        T::coordinate(row.max(column), row.min(column))
    })
}

fn hermitian<T: Coordinate>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, |row, column| {
        let value = T::coordinate(row.max(column), row.min(column));
        if row == column {
            value.real_diagonal()
        } else if row > column {
            value
        } else {
            value.conjugate()
        }
    })
}

fn positive_definite<T: Coordinate>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, |row, column| {
        if row == column {
            T::positive_diagonal(n)
        } else {
            let value = T::small_phase(row.max(column), row.min(column));
            if row > column {
                value
            } else {
                value.conjugate()
            }
        }
    })
}

fn canonical_lower<T: Coordinate>(matrix: &DMatrix<T>, real_diagonal: bool) -> Vec<T> {
    let mut data = Vec::with_capacity(matrix.nrows() * (matrix.nrows() + 1) / 2);
    for column in 0..matrix.ncols() {
        for row in column..matrix.nrows() {
            let value = matrix[(row, column)];
            data.push(if real_diagonal && row == column {
                value.real_diagonal()
            } else {
                value
            });
        }
    }
    data
}

macro_rules! conversion_matrix {
    ($test:ident, $type:ty, $real:ty) => {
        #[test]
        fn $test() {
            let exact = ConversionTolerance::<$real>::new(0.0, 0.0);
            for n in DIMENSIONS {
                for triangle in [Triangle::Lower, Triangle::Upper] {
                    let expected = triangular::<$type>(n, triangle);
                    let full =
                        FullTriangular::try_from_dmatrix(&source::<$type>(n), triangle).unwrap();
                    assert_eq!(full.as_slice(), expected.as_slice());
                    assert_eq!(full.to_dmatrix(), expected);
                    assert_eq!(full.into_dmatrix(), expected);
                }

                let lower_full = triangular::<$type>(n, Triangle::Lower);
                let lower_data = pack_lower_column_major(&lower_full);
                let lower = PackedLower::from_vec(n, lower_data.clone()).unwrap();
                assert_eq!(lower.to_dmatrix().unwrap(), lower_full);
                assert_eq!(
                    PackedLower::try_from_dmatrix(&lower_full, exact)
                        .unwrap()
                        .as_slice(),
                    lower_data
                );
                assert_eq!(
                    PackedLower::from_lower_triangle(&source::<$type>(n))
                        .unwrap()
                        .as_slice(),
                    lower_data
                );
                assert_eq!(
                    PackedLower::from_lower_triangle(&lower.to_dmatrix().unwrap())
                        .unwrap()
                        .as_slice(),
                    lower_data
                );

                let upper_full = triangular::<$type>(n, Triangle::Upper);
                let upper_data = pack_upper_column_major(&upper_full);
                let upper = PackedUpper::from_vec(n, upper_data.clone()).unwrap();
                assert_eq!(upper.to_dmatrix().unwrap(), upper_full);
                assert_eq!(
                    PackedUpper::try_from_dmatrix(&upper_full, exact)
                        .unwrap()
                        .as_slice(),
                    upper_data
                );
                assert_eq!(
                    PackedUpper::from_upper_triangle(&source::<$type>(n))
                        .unwrap()
                        .as_slice(),
                    upper_data
                );
                assert_eq!(
                    PackedUpper::from_upper_triangle(&upper.to_dmatrix().unwrap())
                        .unwrap()
                        .as_slice(),
                    upper_data
                );

                let symmetric_full = symmetric::<$type>(n);
                let symmetric_data = canonical_lower(&symmetric_full, false);
                let packed = PackedSymmetric::from_vec(n, symmetric_data.clone()).unwrap();
                assert_eq!(packed.to_dmatrix(), symmetric_full);
                assert_eq!(
                    PackedSymmetric::try_from_dmatrix(&symmetric_full, exact)
                        .unwrap()
                        .as_slice(),
                    symmetric_data
                );
                assert_eq!(
                    PackedSymmetric::from_lower_triangle(&source::<$type>(n))
                        .unwrap()
                        .as_slice(),
                    symmetric_data
                );
                assert_eq!(
                    PackedSymmetric::from_lower_triangle(&packed.to_dmatrix())
                        .unwrap()
                        .as_slice(),
                    symmetric_data
                );

                let hermitian_full = hermitian::<$type>(n);
                let hermitian_data = canonical_lower(&hermitian_full, true);
                let packed = PackedHermitian::from_vec(n, hermitian_data.clone()).unwrap();
                assert_eq!(packed.to_dmatrix(), hermitian_full);
                assert_eq!(
                    PackedHermitian::try_from_dmatrix(&hermitian_full, exact)
                        .unwrap()
                        .as_slice(),
                    hermitian_data
                );
                assert_eq!(
                    PackedHermitian::from_lower_triangle(&source::<$type>(n))
                        .unwrap()
                        .as_slice(),
                    canonical_lower(&source::<$type>(n), true)
                );
                assert_eq!(
                    PackedHermitian::from_lower_triangle(&packed.to_dmatrix())
                        .unwrap()
                        .as_slice(),
                    hermitian_data
                );

                let spd_full = positive_definite::<$type>(n);
                let spd_data = canonical_lower(&spd_full, true);
                let packed = PackedSPD::from_vec(n, spd_data.clone()).unwrap();
                assert_eq!(packed.to_dmatrix(), spd_full);
                assert_eq!(
                    PackedSPD::try_from_structured_dmatrix(&spd_full, exact)
                        .unwrap()
                        .as_slice(),
                    spd_data
                );
                assert_eq!(
                    PackedSPD::try_from_dmatrix(&spd_full, exact)
                        .unwrap()
                        .as_slice(),
                    spd_data
                );
                assert_eq!(
                    PackedSPD::from_lower_triangle_unchecked_structure(&source::<$type>(n))
                        .unwrap()
                        .as_slice(),
                    canonical_lower(&source::<$type>(n), true)
                );
                assert_eq!(
                    PackedSPD::from_lower_triangle_unchecked_structure(&packed.to_dmatrix())
                        .unwrap()
                        .as_slice(),
                    spd_data
                );
            }
        }
    };
}
conversion_matrix!(conversion_matrix_f32, f32, f32);
conversion_matrix!(conversion_matrix_f64, f64, f64);
conversion_matrix!(conversion_matrix_complex32, Complex32, f32);
conversion_matrix!(conversion_matrix_complex64, Complex64, f64);

macro_rules! view_nonmutation {
    ($test:ident, $type:ty) => {
        #[test]
        fn $test() {
            for n in DIMENSIONS {
                let lower_expected = triangular::<$type>(n, Triangle::Lower);
                let lower_data = pack_lower_column_major(&lower_expected);
                let view = PackedLowerView::from_slice(n, &lower_data).unwrap();
                assert_eq!(view.to_dmatrix().unwrap(), lower_expected);
                let mut lower_backing = lower_data.clone();
                let view = PackedLowerViewMut::from_slice_mut(n, &mut lower_backing).unwrap();
                assert_eq!(view.to_dmatrix().unwrap(), lower_expected);
                assert_eq!(lower_backing, lower_data);

                let upper_expected = triangular::<$type>(n, Triangle::Upper);
                let upper_data = pack_upper_column_major(&upper_expected);
                let view = PackedUpperView::from_slice(n, &upper_data).unwrap();
                assert_eq!(view.to_dmatrix().unwrap(), upper_expected);
                let mut upper_backing = upper_data.clone();
                let view = PackedUpperViewMut::from_slice_mut(n, &mut upper_backing).unwrap();
                assert_eq!(view.to_dmatrix().unwrap(), upper_expected);
                assert_eq!(upper_backing, upper_data);

                let symmetric_expected = symmetric::<$type>(n);
                let symmetric_data = canonical_lower(&symmetric_expected, false);
                let view = PackedSymmetricView::from_slice(n, &symmetric_data).unwrap();
                assert_eq!(view.to_dmatrix(), symmetric_expected);
                let mut symmetric_backing = symmetric_data.clone();
                let view =
                    PackedSymmetricViewMut::from_slice_mut(n, &mut symmetric_backing).unwrap();
                assert_eq!(view.to_dmatrix(), symmetric_expected);
                assert_eq!(symmetric_backing, symmetric_data);

                let hermitian_expected = hermitian::<$type>(n);
                let hermitian_data = canonical_lower(&hermitian_expected, true);
                let view = PackedHermitianView::from_slice(n, &hermitian_data).unwrap();
                assert_eq!(view.to_dmatrix(), hermitian_expected);
                let mut hermitian_backing = hermitian_data.clone();
                let view =
                    PackedHermitianViewMut::from_slice_mut(n, &mut hermitian_backing).unwrap();
                assert_eq!(view.to_dmatrix(), hermitian_expected);
                assert_eq!(hermitian_backing, hermitian_data);

                let spd_expected = positive_definite::<$type>(n);
                let spd_data = canonical_lower(&spd_expected, true);
                let view = PackedSPDView::from_slice(n, &spd_data).unwrap();
                assert_eq!(view.to_dmatrix(), spd_expected);
                let mut spd_backing = spd_data.clone();
                let view = PackedSPDViewMut::from_slice_mut(n, &mut spd_backing).unwrap();
                assert_eq!(view.to_dmatrix(), spd_expected);
                assert_eq!(spd_backing, spd_data);
            }
        }
    };
}
view_nonmutation!(view_nonmutation_f32, f32);
view_nonmutation!(view_nonmutation_f64, f64);
view_nonmutation!(view_nonmutation_complex32, Complex32);
view_nonmutation!(view_nonmutation_complex64, Complex64);

#[test]
fn extraction_ignores_the_documented_triangle_while_strict_conversion_rejects_it() {
    let matrix = DMatrix::from_row_slice(3, 3, &[1.0, 91.0, 92.0, 2.0, 3.0, 93.0, 4.0, 5.0, 6.0]);
    assert_eq!(
        PackedLower::from_lower_triangle(&matrix)
            .unwrap()
            .as_slice(),
        &[1.0, 2.0, 4.0, 3.0, 5.0, 6.0]
    );
    assert_eq!(
        PackedUpper::from_upper_triangle(&matrix)
            .unwrap()
            .as_slice(),
        &[1.0, 91.0, 3.0, 92.0, 93.0, 6.0]
    );
    let exact = ConversionTolerance::new(0.0, 0.0);
    assert!(PackedLower::try_from_dmatrix(&matrix, exact).is_err());
    assert!(PackedUpper::try_from_dmatrix(&matrix, exact).is_err());
}

#[test]
fn full_triangular_copy_and_move_preserve_the_ownership_contract() {
    let data = triangular::<Complex64>(5, Triangle::Lower)
        .as_slice()
        .to_vec();
    let borrowed = FullTriangular::from_vec(5, Triangle::Lower, data.clone()).unwrap();
    assert_eq!(borrowed.to_dmatrix().as_slice(), data);
    assert_eq!(borrowed.as_slice(), data);
    let consumed = FullTriangular::from_vec(5, Triangle::Lower, data.clone()).unwrap();
    assert_eq!(consumed.into_dmatrix().as_slice(), data);
}

proptest! {
    #![proptest_config(property_config())]

    #[test]
    fn symmetric_strict_round_trip_matches_independent_indexing(n in 0usize..12, seed in any::<u64>()) {
        let len = n * (n + 1) / 2;
        let data: Vec<_> = (0..len).map(|i| ((seed.wrapping_add(i as u64) % 1009) as f64 - 504.0) / 31.0).collect();
        let full = DMatrix::from_fn(n, n, |row, column| {
            let r = row.max(column);
            let c = row.min(column);
            data[c * (2 * n - c + 1) / 2 + r - c]
        });
        let packed = PackedSymmetric::try_from_dmatrix(&full, ConversionTolerance::new(0.0, 0.0)).unwrap();
        prop_assert_eq!(packed.as_slice(), data.as_slice());
        prop_assert_eq!(packed.to_dmatrix(), full);
    }

    #[test]
    fn lower_and_upper_arbitrary_storage_round_trip_and_zero_opposite_triangles(n in 0usize..12, seed in any::<u64>()) {
        let len = n * (n + 1) / 2;
        let data: Vec<_> = (0..len).map(|i| ((seed.wrapping_add(i as u64) % 997) as f64) / 29.0).collect();
        let lower = PackedLower::from_vec(n, data.clone()).unwrap();
        let lower_full = lower.to_dmatrix().unwrap();
        for column in 0..n { for row in 0..column { prop_assert_eq!(lower_full[(row, column)], 0.0); } }
        let lower_round_trip = PackedLower::try_from_dmatrix(
            &lower_full,
            ConversionTolerance::new(0.0, 0.0),
        ).unwrap();
        prop_assert_eq!(lower_round_trip.as_slice(), data.as_slice());

        let upper = PackedUpper::from_vec(n, data.clone()).unwrap();
        let upper_full = upper.to_dmatrix().unwrap();
        for column in 0..n { for row in column + 1..n { prop_assert_eq!(upper_full[(row, column)], 0.0); } }
        let upper_round_trip = PackedUpper::try_from_dmatrix(
            &upper_full,
            ConversionTolerance::new(0.0, 0.0),
        ).unwrap();
        prop_assert_eq!(upper_round_trip.as_slice(), data.as_slice());
    }

    #[test]
    fn complex_symmetric_and_hermitian_invariants_stay_distinct(n in 2usize..12, seed in any::<u64>()) {
        let mut data = Vec::with_capacity(n * (n + 1) / 2);
        for column in 0..n { for row in column..n {
            data.push(Complex64::new(((seed.wrapping_add((row * 17 + column) as u64) % 101) as f64) / 17.0, (row + column + 1) as f64 / 13.0));
        }}
        let symmetric = PackedSymmetric::from_vec(n, data.clone()).unwrap().to_dmatrix();
        for column in 0..n { for row in 0..n { prop_assert_eq!(symmetric[(row, column)], symmetric[(column, row)]); } }
        prop_assert_ne!(symmetric[(1, 0)], symmetric[(0, 1)].conj());
        let symmetric_round_trip = PackedSymmetric::try_from_dmatrix(
            &symmetric,
            ConversionTolerance::new(0.0, 0.0),
        ).unwrap();
        prop_assert_eq!(symmetric_round_trip.as_slice(), data.as_slice());
        for index in 0..n { let offset = index * (2 * n - index + 1) / 2; data[offset].im = 0.0; }
        let hermitian = PackedHermitian::from_vec(n, data).unwrap().to_dmatrix();
        for column in 0..n { for row in 0..n { prop_assert_eq!(hermitian[(row, column)], hermitian[(column, row)].conj()); } }
        let hermitian_round_trip = PackedHermitian::try_from_dmatrix(
            &hermitian,
            ConversionTolerance::new(0.0, 0.0),
        ).unwrap();
        prop_assert_eq!(hermitian_round_trip.to_dmatrix(), hermitian);
    }

    #[test]
    fn independent_spd_and_hpd_generators_pass_nalgebra_and_conversion_cholesky(n in 0usize..12, seed in any::<u64>()) {
        let spd = spd_f64(n, seed, 1.0);
        prop_assert!(spd.clone().cholesky().is_some());
        prop_assert!(PackedSPD::try_from_dmatrix(&spd, ConversionTolerance::new(1.0e-12, 1.0e-12)).is_ok());
        let hpd = hpd_complex64(n, seed ^ 0x5a5a_5a5a, 1.0);
        prop_assert!(hpd.clone().cholesky().is_some());
        prop_assert!(PackedSPD::try_from_dmatrix(&hpd, ConversionTolerance::new(1.0e-12, 1.0e-12)).is_ok());
    }

    #[test]
    fn mutable_view_is_unchanged_and_extraction_reads_only_lower(n in 0usize..12, seed in any::<u64>()) {
        let full = spd_f64(n, seed, 1.0);
        let data = pack_lower_column_major(&full);
        let mut backing = data.clone();
        let view = PackedSPDViewMut::from_slice_mut(n, &mut backing).unwrap();
        let mut converted = view.to_dmatrix();
        prop_assert_eq!(backing.as_slice(), data.as_slice());
        for column in 0..n { for row in 0..column { converted[(row, column)] += 1000.0; } }
        let extracted = PackedSPD::from_lower_triangle_unchecked_structure(&converted).unwrap();
        prop_assert_eq!(extracted.as_slice(), data.as_slice());
    }
}
