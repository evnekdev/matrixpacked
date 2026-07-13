use matrixpacked::{
    LapackScalar, PackedHermitian, PackedLower, PackedSPD, PackedSymmetric, PackedUpper,
    storage::PackedStorage,
};
use nalgebra::{DMatrix, Scalar};

use super::compare::OracleScalar;

/// LAPACK lower packed-column layout. Column `j` starts at
/// `j * (2*n - j + 1) / 2`; `(i, j)` is at that offset plus `i - j`.
pub fn pack_lower_column_major<T: Scalar + Clone>(matrix: &DMatrix<T>) -> Vec<T> {
    assert_eq!(matrix.nrows(), matrix.ncols());
    let n = matrix.nrows();
    let mut packed = Vec::with_capacity(n * (n + 1) / 2);
    for col in 0..n {
        for row in col..n {
            packed.push(matrix[(row, col)].clone());
        }
    }
    packed
}

/// LAPACK upper packed-column layout. Column `j` starts at `j * (j + 1) / 2`,
/// and `(i, j)` is at that offset plus `i`.
pub fn pack_upper_column_major<T: Scalar + Clone>(matrix: &DMatrix<T>) -> Vec<T> {
    assert_eq!(matrix.nrows(), matrix.ncols());
    let n = matrix.nrows();
    let mut packed = Vec::with_capacity(n * (n + 1) / 2);
    for col in 0..n {
        for row in 0..=col {
            packed.push(matrix[(row, col)].clone());
        }
    }
    packed
}

pub fn decode_lower_packed<T: OracleScalar>(n: usize, packed: &[T]) -> DMatrix<T> {
    assert_eq!(packed.len(), n * (n + 1) / 2);
    DMatrix::from_fn(n, n, |row, col| {
        if row < col {
            T::zero()
        } else {
            packed[col * (2 * n - col + 1) / 2 + row - col]
        }
    })
}

pub fn decode_upper_packed<T: OracleScalar>(n: usize, packed: &[T]) -> DMatrix<T> {
    assert_eq!(packed.len(), n * (n + 1) / 2);
    DMatrix::from_fn(n, n, |row, col| {
        if row > col {
            T::zero()
        } else {
            packed[col * (col + 1) / 2 + row]
        }
    })
}

pub fn lower_to_dmatrix<T, S>(packed: &PackedLower<T, S>) -> DMatrix<T>
where
    T: OracleScalar + LapackScalar,
    S: PackedStorage<T>,
{
    DMatrix::from_fn(packed.dimension(), packed.dimension(), |row, col| {
        packed.get(row, col).unwrap()
    })
}

pub fn upper_to_dmatrix<T, S>(packed: &PackedUpper<T, S>) -> DMatrix<T>
where
    T: OracleScalar + LapackScalar,
    S: PackedStorage<T>,
{
    DMatrix::from_fn(packed.dimension(), packed.dimension(), |row, col| {
        packed.get(row, col).unwrap()
    })
}

pub fn symmetric_to_dmatrix<T, S>(packed: &PackedSymmetric<T, S>) -> DMatrix<T>
where
    T: OracleScalar + LapackScalar,
    S: PackedStorage<T>,
{
    DMatrix::from_fn(packed.dimension(), packed.dimension(), |row, col| {
        packed.get(row, col).unwrap()
    })
}

pub fn spd_to_dmatrix<T, S>(packed: &PackedSPD<T, S>) -> DMatrix<T>
where
    T: OracleScalar + LapackScalar,
    S: PackedStorage<T>,
{
    DMatrix::from_fn(packed.dimension(), packed.dimension(), |row, col| {
        packed.get(row, col).unwrap()
    })
}

pub fn hermitian_to_dmatrix<T, S>(packed: &PackedHermitian<T, S>) -> DMatrix<T>
where
    T: OracleScalar + LapackScalar,
    S: PackedStorage<T>,
{
    DMatrix::from_fn(packed.dimension(), packed.dimension(), |row, col| {
        packed.get(row, col).unwrap()
    })
}
