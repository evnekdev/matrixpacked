use std::{any::type_name, fmt::Debug};

use nalgebra::{ComplexField, DMatrix, DVector, RealField};
use num_complex::{Complex32, Complex64};

#[derive(Clone, Copy, Debug)]
pub struct Tolerance<R> {
    pub abs: R,
    pub rel: R,
}

impl Tolerance<f64> {
    pub fn for_scalar<T: OracleScalar>() -> Self {
        T::default_tolerance()
    }
}

pub trait OracleScalar: ComplexField + Copy + Debug + Send + Sync + 'static {
    fn scalar_name() -> &'static str;
    fn magnitude(value: Self) -> f64;
    fn real_to_f64(value: Self::RealField) -> f64;
    fn default_tolerance() -> Tolerance<f64>;
}

macro_rules! impl_real_oracle_scalar {
    ($ty:ty, $abs:expr, $rel:expr) => {
        impl OracleScalar for $ty {
            fn scalar_name() -> &'static str {
                stringify!($ty)
            }
            fn magnitude(value: Self) -> f64 {
                value.abs() as f64
            }
            fn real_to_f64(value: Self::RealField) -> f64 {
                value as f64
            }
            fn default_tolerance() -> Tolerance<f64> {
                Tolerance {
                    abs: $abs,
                    rel: $rel,
                }
            }
        }
    };
}

impl_real_oracle_scalar!(f32, 2.0e-5, 2.0e-4);
impl_real_oracle_scalar!(f64, 2.0e-12, 2.0e-11);

impl OracleScalar for Complex32 {
    fn scalar_name() -> &'static str {
        "Complex32"
    }
    fn magnitude(value: Self) -> f64 {
        value.norm() as f64
    }
    fn real_to_f64(value: Self::RealField) -> f64 {
        value as f64
    }
    fn default_tolerance() -> Tolerance<f64> {
        Tolerance {
            abs: 3.0e-5,
            rel: 3.0e-4,
        }
    }
}

impl OracleScalar for Complex64 {
    fn scalar_name() -> &'static str {
        "Complex64"
    }
    fn magnitude(value: Self) -> f64 {
        value.norm()
    }
    fn real_to_f64(value: Self::RealField) -> f64 {
        value
    }
    fn default_tolerance() -> Tolerance<f64> {
        Tolerance {
            abs: 3.0e-12,
            rel: 3.0e-11,
        }
    }
}

pub fn assert_scalar_close<T: OracleScalar>(
    expected: T,
    actual: T,
    tolerance: Tolerance<f64>,
    dimension: impl std::fmt::Display,
    index: impl std::fmt::Display,
) {
    let absolute_error = T::magnitude(actual - expected);
    let scale = T::magnitude(actual).max(T::magnitude(expected));
    let relative_error = if scale == 0.0 {
        0.0
    } else {
        absolute_error / scale
    };
    let threshold = tolerance.abs + tolerance.rel * scale;
    assert!(
        absolute_error <= threshold,
        "oracle comparison failed: dimension={dimension}, scalar={}, index={index}, expected={expected:?}, actual={actual:?}, absolute_error={absolute_error:e}, relative_error={relative_error:e}, threshold={threshold:e} (abs_tol={:e}, rel_tol={:e})",
        T::scalar_name(),
        tolerance.abs,
        tolerance.rel,
    );
}

pub fn assert_slice_close<T: OracleScalar>(
    expected: &[T],
    actual: &[T],
    tolerance: Tolerance<f64>,
    dimension: usize,
) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "slice length mismatch for dimension={dimension}, scalar={}",
        T::scalar_name()
    );
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        assert_scalar_close(expected, actual, tolerance, dimension, index);
    }
}

pub fn assert_matrix_close<T: OracleScalar>(
    expected: &DMatrix<T>,
    actual: &DMatrix<T>,
    tolerance: Tolerance<f64>,
) {
    assert_eq!(
        expected.shape(),
        actual.shape(),
        "matrix shape mismatch: expected={:?}, actual={:?}, scalar={}",
        expected.shape(),
        actual.shape(),
        T::scalar_name()
    );
    let dimension = format!("{}x{}", expected.nrows(), expected.ncols());
    for row in 0..expected.nrows() {
        for col in 0..expected.ncols() {
            let absolute_error = T::magnitude(actual[(row, col)] - expected[(row, col)]);
            let scale = T::magnitude(actual[(row, col)]).max(T::magnitude(expected[(row, col)]));
            if absolute_error > tolerance.abs + tolerance.rel * scale {
                let difference = actual - expected;
                let residual = T::real_to_f64(difference.norm());
                let expected_norm = T::real_to_f64(expected.norm());
                let normalized_residual = residual / expected_norm.max(f64::EPSILON);
                panic!(
                    "oracle matrix comparison failed: dimension={dimension}, scalar={}, index=({row}, {col}), expected={:?}, actual={:?}, absolute_error={absolute_error:e}, relative_error={:e}, matrix_residual={residual:e}, normalized_matrix_residual={normalized_residual:e}",
                    T::scalar_name(),
                    expected[(row, col)],
                    actual[(row, col)],
                    absolute_error / scale.max(f64::EPSILON),
                );
            }
        }
    }
}

pub fn assert_identity_close<T: OracleScalar>(matrix: &DMatrix<T>, tolerance: Tolerance<f64>) {
    assert_eq!(
        matrix.nrows(),
        matrix.ncols(),
        "identity check requires a square matrix"
    );
    assert_matrix_close(
        &DMatrix::identity(matrix.nrows(), matrix.ncols()),
        matrix,
        tolerance,
    );
}

pub fn assert_hermitian<T: OracleScalar>(matrix: &DMatrix<T>, tolerance: Tolerance<f64>) {
    assert_eq!(
        matrix.nrows(),
        matrix.ncols(),
        "Hermitian check requires a square matrix"
    );
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            assert_scalar_close(
                matrix[(row, col)],
                matrix[(col, row)].conjugate(),
                tolerance,
                matrix.nrows(),
                format_args!("({row}, {col})"),
            );
        }
    }
}

pub fn assert_symmetric<T: OracleScalar>(matrix: &DMatrix<T>, tolerance: Tolerance<f64>) {
    assert_eq!(
        matrix.nrows(),
        matrix.ncols(),
        "symmetric check requires a square matrix"
    );
    assert_matrix_close(matrix, &matrix.transpose(), tolerance);
}

pub fn assert_unitary<T: OracleScalar>(matrix: &DMatrix<T>, tolerance: Tolerance<f64>) {
    assert_identity_close(&(matrix.adjoint() * matrix), tolerance);
}

pub fn assert_orthogonal<T: OracleScalar<RealField = T>>(
    matrix: &DMatrix<T>,
    tolerance: Tolerance<f64>,
) where
    T: RealField,
{
    assert_identity_close(&(matrix.transpose() * matrix), tolerance);
}

pub fn vector_residual<T: OracleScalar>(a: &DMatrix<T>, x: &DVector<T>, b: &DVector<T>) -> f64 {
    let residual = a * x - b;
    T::real_to_f64(residual.norm())
        / (T::real_to_f64(a.norm()) * T::real_to_f64(x.norm()) + T::real_to_f64(b.norm()))
            .max(f64::EPSILON)
}

pub fn matrix_rhs_residual<T: OracleScalar>(a: &DMatrix<T>, x: &DMatrix<T>, b: &DMatrix<T>) -> f64 {
    let residual = a * x - b;
    T::real_to_f64(residual.norm())
        / (T::real_to_f64(a.norm()) * T::real_to_f64(x.norm()) + T::real_to_f64(b.norm()))
            .max(f64::EPSILON)
}

pub fn inverse_residual<T: OracleScalar>(a: &DMatrix<T>, inverse: &DMatrix<T>) -> f64 {
    let identity = DMatrix::identity(a.nrows(), a.ncols());
    matrix_rhs_residual(a, inverse, &identity)
}

pub fn eigen_residual<T: OracleScalar>(a: &DMatrix<T>, vector: &DVector<T>, eigenvalue: T) -> f64 {
    let rhs = vector * eigenvalue;
    vector_residual(a, vector, &rhs)
}

pub fn generalized_eigen_residual<T: OracleScalar>(
    a: &DMatrix<T>,
    b: &DMatrix<T>,
    vector: &DVector<T>,
    eigenvalue: T,
) -> f64 {
    let rhs = (b * vector) * eigenvalue;
    vector_residual(a, vector, &rhs)
}

pub fn scalar_type_name<T>() -> &'static str {
    type_name::<T>()
}
