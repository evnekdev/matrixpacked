#![cfg(feature = "nalgebra-interop")]

use matrixpacked::{FullTriangular, PackedMatrixError, Triangle};
use nalgebra::DMatrix;
use num_complex::Complex64;

fn real_data(n: usize) -> Vec<f64> {
    (0..n)
        .flat_map(|column| (0..n).map(move |row| (100 * column + row) as f64 + 1.0))
        .collect()
}

fn triangular_data(n: usize, triangle: Triangle) -> Vec<f64> {
    let mut data = real_data(n);
    for column in 0..n {
        for row in 0..n {
            if matches!(triangle, Triangle::Lower) && row < column
                || matches!(triangle, Triangle::Upper) && row > column
            {
                data[column * n + row] = 0.0;
            }
        }
    }
    data
}

#[test]
fn lower_to_dmatrix_preserves_source_and_column_major_order() {
    let data = triangular_data(3, Triangle::Lower);
    let full = FullTriangular::from_vec(3, Triangle::Lower, data.clone()).unwrap();

    let matrix = full.to_dmatrix();

    assert_eq!(full.as_slice(), data);
    assert_eq!(matrix.shape(), (3, 3));
    assert_eq!(matrix.as_slice(), data);
    assert_eq!(matrix[(2, 1)], 103.0);
}

#[test]
fn upper_into_dmatrix_moves_correct_data() {
    let data = triangular_data(3, Triangle::Upper);
    let full = FullTriangular::from_vec(3, Triangle::Upper, data.clone()).unwrap();

    let matrix: DMatrix<f64> = full.into();

    assert_eq!(matrix.shape(), (3, 3));
    assert_eq!(matrix.as_slice(), data);
    assert_eq!(matrix[(1, 2)], 202.0);
}

#[test]
fn complex_values_round_trip() {
    let data: Vec<_> = (0..3)
        .flat_map(|column| {
            (0..3).map(move |row| {
                if row >= column {
                    Complex64::new((100 * column + row) as f64, (10 * row + column) as f64)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
        })
        .collect();
    let full = FullTriangular::from_vec(3, Triangle::Lower, data.clone()).unwrap();

    let matrix = full.to_dmatrix();
    let converted = FullTriangular::try_from_dmatrix(&matrix, Triangle::Lower).unwrap();

    assert_eq!(matrix.as_slice(), data);
    assert_eq!(converted.as_slice(), data);
}

#[test]
fn dmatrix_to_lower_copies_selected_triangle_and_zeros_the_other() {
    let matrix = DMatrix::from_vec(3, 3, real_data(3));

    let full = FullTriangular::try_from_dmatrix(&matrix, Triangle::Lower).unwrap();

    assert_eq!(full.triangle(), Triangle::Lower);
    assert_eq!(full.as_slice(), triangular_data(3, Triangle::Lower));
    assert_eq!(matrix.as_slice(), real_data(3));
}

#[test]
fn dmatrix_to_upper_copies_selected_triangle_and_zeros_the_other() {
    let matrix = DMatrix::from_vec(3, 3, real_data(3));

    let full = FullTriangular::try_from_dmatrix(&matrix, Triangle::Upper).unwrap();

    assert_eq!(full.triangle(), Triangle::Upper);
    assert_eq!(full.as_slice(), triangular_data(3, Triangle::Upper));
}

#[test]
fn non_square_dmatrix_is_rejected() {
    let matrix = DMatrix::from_vec(2, 3, (0..6).map(f64::from).collect());

    assert_eq!(
        FullTriangular::try_from_dmatrix(&matrix, Triangle::Lower),
        Err(PackedMatrixError::NonSquareMatrix {
            rows: 2,
            columns: 3,
        })
    );
}

#[test]
fn dimensions_preserve_shape_and_exact_storage_order() {
    for n in [0, 1, 2, 3, 4, 5, 8, 9] {
        for triangle in [Triangle::Lower, Triangle::Upper] {
            let matrix = DMatrix::from_vec(n, n, real_data(n));
            let full = FullTriangular::try_from_dmatrix(&matrix, triangle).unwrap();
            let expected = triangular_data(n, triangle);

            assert_eq!(full.dimension(), n);
            assert_eq!(full.as_slice(), expected);
            assert_eq!(full.to_dmatrix().shape(), (n, n));
            assert_eq!(full.into_dmatrix().as_slice(), expected);
        }
    }
}
