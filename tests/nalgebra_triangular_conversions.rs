#![cfg(feature = "nalgebra-interop")]

use matrixpacked::{
    PackedLower, PackedLowerView, PackedLowerViewMut, PackedMatrixError, PackedUpper,
    PackedUpperView, PackedUpperViewMut,
};
use nalgebra::{DMatrix, Scalar};
use num_complex::{Complex32, Complex64};
use num_traits::Zero;

trait Coordinate: Scalar + Copy + Zero {
    fn coordinate(row: usize, column: usize) -> Self;
}

macro_rules! impl_real_coordinate {
    ($type:ty) => {
        impl Coordinate for $type {
            fn coordinate(row: usize, column: usize) -> Self {
                (100 * column + row + 1) as Self
            }
        }
    };
}

impl_real_coordinate!(f32);
impl_real_coordinate!(f64);

impl Coordinate for Complex32 {
    fn coordinate(row: usize, column: usize) -> Self {
        Self::new(
            (100 * column + row + 1) as f32,
            (10 * row + column + 1) as f32,
        )
    }
}

impl Coordinate for Complex64 {
    fn coordinate(row: usize, column: usize) -> Self {
        Self::new(
            (100 * column + row + 1) as f64,
            (10 * row + column + 1) as f64,
        )
    }
}

fn source_matrix<T: Coordinate>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, T::coordinate)
}

fn expected_lower_matrix<T: Coordinate>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, |row, column| {
        if row >= column {
            T::coordinate(row, column)
        } else {
            T::zero()
        }
    })
}

fn expected_upper_matrix<T: Coordinate>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, |row, column| {
        if row <= column {
            T::coordinate(row, column)
        } else {
            T::zero()
        }
    })
}

fn expected_lower_packed<T: Coordinate>(n: usize) -> Vec<T> {
    let mut packed = Vec::with_capacity(n * (n + 1) / 2);
    for column in 0..n {
        for row in column..n {
            packed.push(T::coordinate(row, column));
        }
    }
    packed
}

fn expected_upper_packed<T: Coordinate>(n: usize) -> Vec<T> {
    let mut packed = Vec::with_capacity(n * (n + 1) / 2);
    for column in 0..n {
        for row in 0..=column {
            packed.push(T::coordinate(row, column));
        }
    }
    packed
}

macro_rules! conversion_tests {
    ($test:ident, $type:ty) => {
        #[test]
        fn $test() {
            for n in [0, 1, 2, 3, 4, 7] {
                let source = source_matrix::<$type>(n);
                let expected_lower = expected_lower_matrix::<$type>(n);
                let expected_upper = expected_upper_matrix::<$type>(n);
                let lower_data = expected_lower_packed::<$type>(n);
                let upper_data = expected_upper_packed::<$type>(n);

                let lower = PackedLower::from_vec(n, lower_data.clone()).unwrap();
                let upper = PackedUpper::from_vec(n, upper_data.clone()).unwrap();
                assert_eq!(lower.to_dmatrix().unwrap(), expected_lower);
                assert_eq!(upper.to_dmatrix().unwrap(), expected_upper);

                let extracted_lower = PackedLower::from_lower_triangle(&source).unwrap();
                let extracted_upper = PackedUpper::from_upper_triangle(&source).unwrap();
                assert_eq!(source, source_matrix::<$type>(n));
                assert_eq!(extracted_lower.as_slice(), lower_data);
                assert_eq!(extracted_upper.as_slice(), upper_data);
                assert_eq!(extracted_lower.to_dmatrix().unwrap(), expected_lower);
                assert_eq!(extracted_upper.to_dmatrix().unwrap(), expected_upper);

                let lower_round_trip =
                    PackedLower::from_lower_triangle(&lower.to_dmatrix().unwrap()).unwrap();
                let upper_round_trip =
                    PackedUpper::from_upper_triangle(&upper.to_dmatrix().unwrap()).unwrap();
                assert_eq!(lower_round_trip.as_slice(), lower_data);
                assert_eq!(upper_round_trip.as_slice(), upper_data);
            }
        }
    };
}

conversion_tests!(f32_conversions, f32);
conversion_tests!(f64_conversions, f64);
conversion_tests!(complex32_conversions, Complex32);
conversion_tests!(complex64_conversions, Complex64);

#[test]
fn immutable_and_mutable_views_are_not_modified() {
    let lower_data = expected_lower_packed::<f64>(3);
    let upper_data = expected_upper_packed::<f64>(3);

    let lower_view = PackedLowerView::from_slice(3, &lower_data).unwrap();
    let upper_view = PackedUpperView::from_slice(3, &upper_data).unwrap();
    assert_eq!(
        lower_view.to_dmatrix().unwrap(),
        expected_lower_matrix::<f64>(3)
    );
    assert_eq!(
        upper_view.to_dmatrix().unwrap(),
        expected_upper_matrix::<f64>(3)
    );
    assert_eq!(lower_view.as_slice(), lower_data);
    assert_eq!(upper_view.as_slice(), upper_data);

    let mut lower_mut_data = lower_data.clone();
    let mut upper_mut_data = upper_data.clone();
    {
        let lower_view = PackedLowerViewMut::from_slice_mut(3, &mut lower_mut_data).unwrap();
        let upper_view = PackedUpperViewMut::from_slice_mut(3, &mut upper_mut_data).unwrap();
        assert_eq!(
            lower_view.to_dmatrix().unwrap(),
            expected_lower_matrix::<f64>(3)
        );
        assert_eq!(
            upper_view.to_dmatrix().unwrap(),
            expected_upper_matrix::<f64>(3)
        );
    }
    assert_eq!(lower_mut_data, lower_data);
    assert_eq!(upper_mut_data, upper_data);
}

#[test]
fn non_square_matrices_are_rejected() {
    let matrix = DMatrix::from_element(2, 3, 1.0_f64);
    let error = PackedMatrixError::NonSquareMatrix {
        rows: 2,
        columns: 3,
    };

    assert_eq!(
        PackedLower::from_lower_triangle(&matrix).unwrap_err(),
        error.clone()
    );
    assert_eq!(
        PackedUpper::from_upper_triangle(&matrix).unwrap_err(),
        error
    );
}
