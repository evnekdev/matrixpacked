use nalgebra::{ComplexField, DMatrix, DVector};
use num_complex::{Complex32, Complex64};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub trait GeneratedScalar: ComplexField + Copy + std::fmt::Debug + 'static {
    fn sample(rng: &mut ChaCha8Rng) -> Self;
    fn from_f64(value: f64) -> Self;
    fn real_part(value: Self) -> Self;
    fn diagonal_with_magnitude(value: Self, minimum: f64) -> Self;
}

macro_rules! impl_real_generated_scalar {
    ($ty:ty) => {
        impl GeneratedScalar for $ty {
            fn sample(rng: &mut ChaCha8Rng) -> Self {
                rng.random_range(-1.0..=1.0) as $ty
            }
            fn from_f64(value: f64) -> Self {
                value as $ty
            }
            fn real_part(value: Self) -> Self {
                value
            }
            fn diagonal_with_magnitude(value: Self, minimum: f64) -> Self {
                let minimum = minimum as $ty;
                if value >= 0.0 {
                    value + minimum
                } else {
                    value - minimum
                }
            }
        }
    };
}
impl_real_generated_scalar!(f32);
impl_real_generated_scalar!(f64);

impl GeneratedScalar for Complex32 {
    fn sample(rng: &mut ChaCha8Rng) -> Self {
        Self::new(rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0))
    }
    fn from_f64(value: f64) -> Self {
        Self::new(value as f32, 0.0)
    }
    fn real_part(value: Self) -> Self {
        Self::new(value.re, 0.0)
    }
    fn diagonal_with_magnitude(value: Self, minimum: f64) -> Self {
        if value.norm() < minimum as f32 {
            value + Self::new(minimum as f32, 0.0)
        } else {
            value
        }
    }
}

impl GeneratedScalar for Complex64 {
    fn sample(rng: &mut ChaCha8Rng) -> Self {
        Self::new(rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0))
    }
    fn from_f64(value: f64) -> Self {
        Self::new(value, 0.0)
    }
    fn real_part(value: Self) -> Self {
        Self::new(value.re, 0.0)
    }
    fn diagonal_with_magnitude(value: Self, minimum: f64) -> Self {
        if value.norm() < minimum {
            value + Self::new(minimum, 0.0)
        } else {
            value
        }
    }
}

fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

fn arbitrary_matrix<T: GeneratedScalar>(rows: usize, cols: usize, seed: u64) -> DMatrix<T> {
    let mut rng = rng(seed);
    DMatrix::from_fn(rows, cols, |_, _| T::sample(&mut rng))
}

pub fn arbitrary_lower<T: GeneratedScalar>(n: usize, seed: u64) -> DMatrix<T> {
    let mut rng = rng(seed);
    DMatrix::from_fn(n, n, |row, col| {
        if row >= col {
            T::sample(&mut rng)
        } else {
            T::zero()
        }
    })
}

pub fn arbitrary_upper<T: GeneratedScalar>(n: usize, seed: u64) -> DMatrix<T> {
    let mut rng = rng(seed);
    DMatrix::from_fn(n, n, |row, col| {
        if row <= col {
            T::sample(&mut rng)
        } else {
            T::zero()
        }
    })
}

pub fn nonsingular_lower<T: GeneratedScalar>(
    n: usize,
    seed: u64,
    minimum_diagonal: f64,
) -> DMatrix<T> {
    let mut matrix = arbitrary_lower(n, seed);
    for index in 0..n {
        matrix[(index, index)] =
            T::diagonal_with_magnitude(matrix[(index, index)], minimum_diagonal);
    }
    matrix
}

pub fn nonsingular_upper<T: GeneratedScalar>(
    n: usize,
    seed: u64,
    minimum_diagonal: f64,
) -> DMatrix<T> {
    let mut matrix = arbitrary_upper(n, seed);
    for index in 0..n {
        matrix[(index, index)] =
            T::diagonal_with_magnitude(matrix[(index, index)], minimum_diagonal);
    }
    matrix
}

pub fn unit_lower<T: GeneratedScalar>(n: usize, seed: u64) -> DMatrix<T> {
    let mut matrix = arbitrary_lower(n, seed);
    for index in 0..n {
        matrix[(index, index)] = T::one();
    }
    matrix
}

pub fn unit_upper<T: GeneratedScalar>(n: usize, seed: u64) -> DMatrix<T> {
    let mut matrix = arbitrary_upper(n, seed);
    for index in 0..n {
        matrix[(index, index)] = T::one();
    }
    matrix
}

pub fn real_symmetric_f32(n: usize, seed: u64) -> DMatrix<f32> {
    symmetric_from_lower(arbitrary_lower(n, seed))
}
pub fn real_symmetric_f64(n: usize, seed: u64) -> DMatrix<f64> {
    symmetric_from_lower(arbitrary_lower(n, seed))
}
pub fn complex_symmetric<T: GeneratedScalar>(n: usize, seed: u64) -> DMatrix<T> {
    symmetric_from_lower(arbitrary_lower(n, seed))
}

fn symmetric_from_lower<T: GeneratedScalar>(lower: DMatrix<T>) -> DMatrix<T> {
    DMatrix::from_fn(lower.nrows(), lower.ncols(), |row, col| {
        lower[(row.max(col), row.min(col))]
    })
}

pub fn hermitian<T: GeneratedScalar>(n: usize, seed: u64) -> DMatrix<T> {
    let lower = arbitrary_lower::<T>(n, seed);
    DMatrix::from_fn(n, n, |row, col| {
        if row == col {
            T::real_part(lower[(row, col)])
        } else if row > col {
            lower[(row, col)]
        } else {
            lower[(col, row)].conjugate()
        }
    })
}

pub fn spd_f32(n: usize, seed: u64, shift: f32) -> DMatrix<f32> {
    let m = arbitrary_matrix::<f32>(n, n, seed);
    m.transpose() * &m + DMatrix::identity(n, n) * shift
}

pub fn spd_f64(n: usize, seed: u64, shift: f64) -> DMatrix<f64> {
    let m = arbitrary_matrix::<f64>(n, n, seed);
    m.transpose() * &m + DMatrix::identity(n, n) * shift
}

pub fn well_conditioned_spd_f64(n: usize, seed: u64) -> DMatrix<f64> {
    spd_f64(n, seed, n.max(1) as f64)
}

pub fn moderately_conditioned_spd_f64(n: usize, seed: u64) -> DMatrix<f64> {
    let q = arbitrary_matrix::<f64>(n, n, seed).qr().q();
    let diagonal = DVector::from_fn(n, |index, _| {
        if n <= 1 {
            1.0
        } else {
            10f64.powf(4.0 * index as f64 / (n - 1) as f64)
        }
    });
    &q * DMatrix::from_diagonal(&diagonal) * q.transpose()
}

pub fn deliberately_ill_conditioned_spd_f64(n: usize, seed: u64) -> DMatrix<f64> {
    let q = arbitrary_matrix::<f64>(n, n, seed).qr().q();
    let diagonal = DVector::from_fn(n, |index, _| {
        if n <= 1 {
            1.0
        } else {
            10f64.powf(-10.0 * index as f64 / (n - 1) as f64)
        }
    });
    &q * DMatrix::from_diagonal(&diagonal) * q.transpose()
}

pub fn singular_psd_f64(n: usize, seed: u64) -> DMatrix<f64> {
    if n == 0 {
        return DMatrix::zeros(0, 0);
    }
    let mut m = arbitrary_matrix::<f64>(n, n, seed);
    m.set_column(n - 1, &DVector::zeros(n));
    m.transpose() * m
}

pub fn hpd_complex32(n: usize, seed: u64, shift: f32) -> DMatrix<Complex32> {
    let m = arbitrary_matrix::<Complex32>(n, n, seed);
    m.adjoint() * &m + DMatrix::identity(n, n) * Complex32::new(shift, 0.0)
}

pub fn hpd_complex64(n: usize, seed: u64, shift: f64) -> DMatrix<Complex64> {
    let m = arbitrary_matrix::<Complex64>(n, n, seed);
    m.adjoint() * &m + DMatrix::identity(n, n) * Complex64::new(shift, 0.0)
}

pub fn symmetric_indefinite_f64(n: usize, seed: u64) -> DMatrix<f64> {
    assert!(n >= 2);
    let q = arbitrary_matrix::<f64>(n, n, seed).qr().q();
    let d = DMatrix::from_diagonal(&DVector::from_fn(n, |i, _| {
        if i % 2 == 0 {
            1.0 + i as f64
        } else {
            -(1.0 + i as f64)
        }
    }));
    &q * d * q.transpose()
}

pub fn hermitian_indefinite(n: usize, seed: u64) -> DMatrix<Complex64> {
    assert!(n >= 2);
    let q = arbitrary_matrix::<Complex64>(n, n, seed).qr().q();
    let d = DMatrix::from_diagonal(&DVector::from_fn(n, |i, _| {
        Complex64::new(
            if i % 2 == 0 {
                1.0 + i as f64
            } else {
                -(1.0 + i as f64)
            },
            0.0,
        )
    }));
    &q * d * q.adjoint()
}

pub fn vector<T: GeneratedScalar>(n: usize, seed: u64) -> DVector<T> {
    let mut rng = rng(seed);
    DVector::from_fn(n, |_, _| T::sample(&mut rng))
}

/// A dense matrix's backing slice is column-major, matching LAPACK multi-RHS buffers.
pub fn column_major_multi_rhs<T: GeneratedScalar>(n: usize, nrhs: usize, seed: u64) -> Vec<T> {
    arbitrary_matrix::<T>(n, nrhs, seed).as_slice().to_vec()
}
