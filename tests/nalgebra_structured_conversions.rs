#![cfg(feature = "nalgebra-interop")]

use matrixpacked::{
    PackedHermitian, PackedHermitianView, PackedHermitianViewMut, PackedMatrixError, PackedSPD,
    PackedSPDView, PackedSPDViewMut, PackedSymmetric, PackedSymmetricView, PackedSymmetricViewMut,
};
use nalgebra::{DMatrix, Scalar};
use num_complex::{Complex32, Complex64};
use num_traits::Zero;

trait Coordinate: Scalar + Copy + Zero {
    fn coordinate(row: usize, column: usize) -> Self;
    fn conjugate(self) -> Self;
    fn real_diagonal(self) -> Self;
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
        }
    };
}

impl_complex_coordinate!(Complex32, f32);
impl_complex_coordinate!(Complex64, f64);

fn source_matrix<T: Coordinate>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, T::coordinate)
}

fn lower_data<T: Coordinate>(n: usize, real_diagonal: bool) -> Vec<T> {
    let mut packed = Vec::with_capacity(n * (n + 1) / 2);
    for column in 0..n {
        for row in column..n {
            let value = T::coordinate(row, column);
            packed.push(if real_diagonal && row == column {
                value.real_diagonal()
            } else {
                value
            });
        }
    }
    packed
}

fn expected_symmetric<T: Coordinate>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, |row, column| {
        T::coordinate(row.max(column), row.min(column))
    })
}

fn expected_hermitian<T: Coordinate>(n: usize) -> DMatrix<T> {
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

macro_rules! symmetric_tests {
    ($test:ident, $type:ty) => {
        #[test]
        fn $test() {
            for n in [0, 1, 2, 3, 4, 7] {
                let packed_data = lower_data::<$type>(n, false);
                let packed = PackedSymmetric::from_vec(n, packed_data.clone()).unwrap();
                assert_eq!(packed.to_dmatrix(), expected_symmetric::<$type>(n));

                let source = source_matrix::<$type>(n);
                let extracted = PackedSymmetric::from_lower_triangle(&source).unwrap();
                assert_eq!(source, source_matrix::<$type>(n));
                assert_eq!(extracted.as_slice(), packed_data);
                assert_eq!(extracted.to_dmatrix(), expected_symmetric::<$type>(n));
            }
        }
    };
}

symmetric_tests!(symmetric_f32, f32);
symmetric_tests!(symmetric_f64, f64);
symmetric_tests!(symmetric_complex32, Complex32);
symmetric_tests!(symmetric_complex64, Complex64);

macro_rules! hermitian_tests {
    ($test:ident, $type:ty) => {
        #[test]
        fn $test() {
            for n in [0, 1, 2, 3, 4, 7] {
                let raw_data = lower_data::<$type>(n, false);
                let canonical_data = lower_data::<$type>(n, true);
                let packed = PackedHermitian::from_vec(n, raw_data).unwrap();
                assert_eq!(packed.to_dmatrix(), expected_hermitian::<$type>(n));

                let source = source_matrix::<$type>(n);
                let extracted = PackedHermitian::from_lower_triangle(&source).unwrap();
                assert_eq!(source, source_matrix::<$type>(n));
                assert_eq!(extracted.as_slice(), canonical_data);
                assert_eq!(extracted.to_dmatrix(), expected_hermitian::<$type>(n));
            }
        }
    };
}

hermitian_tests!(hermitian_complex32, Complex32);
hermitian_tests!(hermitian_complex64, Complex64);

macro_rules! spd_tests {
    ($test:ident, $type:ty) => {
        #[test]
        fn $test() {
            for n in [0, 1, 2, 3, 4, 7] {
                let raw_data = lower_data::<$type>(n, false);
                let canonical_data = lower_data::<$type>(n, true);
                let packed = PackedSPD::from_vec(n, raw_data).unwrap();
                assert_eq!(packed.to_dmatrix(), expected_hermitian::<$type>(n));

                let source = source_matrix::<$type>(n);
                let extracted =
                    PackedSPD::from_lower_triangle_unchecked_structure(&source).unwrap();
                assert_eq!(source, source_matrix::<$type>(n));
                assert_eq!(extracted.as_slice(), canonical_data);
                assert_eq!(extracted.to_dmatrix(), expected_hermitian::<$type>(n));
            }
        }
    };
}

spd_tests!(spd_f32, f32);
spd_tests!(spd_f64, f64);
spd_tests!(hpd_complex32, Complex32);
spd_tests!(hpd_complex64, Complex64);

#[test]
fn complex_symmetric_and_hermitian_reconstruction_are_distinct() {
    let c = Complex64::new;
    let data = vec![c(1.0, 7.0), c(2.0, 3.0), c(4.0, 9.0)];
    let symmetric = PackedSymmetric::from_vec(2, data.clone())
        .unwrap()
        .to_dmatrix();
    let hermitian = PackedHermitian::from_vec(2, data).unwrap().to_dmatrix();

    assert_eq!(symmetric[(0, 1)], c(2.0, 3.0));
    assert_eq!(symmetric[(1, 0)], c(2.0, 3.0));
    assert_eq!(symmetric[(0, 0)], c(1.0, 7.0));
    assert_eq!(hermitian[(0, 1)], c(2.0, -3.0));
    assert_eq!(hermitian[(1, 0)], c(2.0, 3.0));
    assert_eq!(hermitian[(0, 0)], c(1.0, 0.0));
}

#[test]
fn owned_and_borrowed_storage_produce_the_same_complete_matrices() {
    let symmetric_data = lower_data::<Complex64>(3, false);
    let expected_symmetric = expected_symmetric::<Complex64>(3);
    let symmetric_view = PackedSymmetricView::from_slice(3, &symmetric_data).unwrap();
    assert_eq!(symmetric_view.to_dmatrix(), expected_symmetric);
    let mut symmetric_mut_data = symmetric_data.clone();
    let symmetric_mut = PackedSymmetricViewMut::from_slice_mut(3, &mut symmetric_mut_data).unwrap();
    assert_eq!(symmetric_mut.to_dmatrix(), expected_symmetric);
    assert_eq!(symmetric_mut_data, symmetric_data);

    let hermitian_data = lower_data::<Complex64>(3, false);
    let expected_hermitian_matrix = expected_hermitian::<Complex64>(3);
    let hermitian_view = PackedHermitianView::from_slice(3, &hermitian_data).unwrap();
    assert_eq!(hermitian_view.to_dmatrix(), expected_hermitian_matrix);
    let mut hermitian_mut_data = hermitian_data.clone();
    let hermitian_mut = PackedHermitianViewMut::from_slice_mut(3, &mut hermitian_mut_data).unwrap();
    assert_eq!(hermitian_mut.to_dmatrix(), expected_hermitian_matrix);
    assert_eq!(hermitian_mut_data, hermitian_data);

    let spd_data = lower_data::<Complex64>(3, false);
    let expected_hpd = expected_hermitian::<Complex64>(3);
    let spd_view = PackedSPDView::from_slice(3, &spd_data).unwrap();
    assert_eq!(spd_view.to_dmatrix(), expected_hpd);
    let mut spd_mut_data = spd_data.clone();
    let spd_mut = PackedSPDViewMut::from_slice_mut(3, &mut spd_mut_data).unwrap();
    assert_eq!(spd_mut.to_dmatrix(), expected_hpd);
    assert_eq!(spd_mut_data, spd_data);
}

#[test]
fn extraction_rejects_non_square_matrices() {
    let matrix = DMatrix::from_element(2, 3, 1.0_f64);
    let expected = PackedMatrixError::NonSquareMatrix {
        rows: 2,
        columns: 3,
    };

    assert_eq!(
        PackedSymmetric::from_lower_triangle(&matrix).unwrap_err(),
        expected
    );
    assert_eq!(
        PackedHermitian::from_lower_triangle(&matrix).unwrap_err(),
        expected
    );
    assert_eq!(
        PackedSPD::from_lower_triangle_unchecked_structure(&matrix).unwrap_err(),
        expected
    );
}
