use matrixpacked::{
    Diagonal, LapackScalar, PackedHermitian, PackedLower, PackedMatrixError, PackedSPD,
    PackedSymmetric, PackedUpper, Transpose,
};
use nalgebra::{DMatrix, DVector};
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use proptest::prelude::*;

use crate::oracle::{
    compare::{
        OracleScalar, Tolerance, assert_hermitian, assert_matrix_close, assert_slice_close,
        assert_symmetric,
    },
    convert::{
        hermitian_to_dmatrix, lower_to_dmatrix, pack_lower_column_major, pack_upper_column_major,
        spd_to_dmatrix, symmetric_to_dmatrix, upper_to_dmatrix,
    },
    generate::{
        GeneratedScalar, arbitrary_lower, arbitrary_upper, hermitian, hpd_complex32, hpd_complex64,
        real_symmetric_f32, real_symmetric_f64, spd_f32, spd_f64, vector,
    },
};

const DIMENSIONS: [usize; 6] = [0, 1, 2, 3, 5, 8];

trait ArithmeticScalar: OracleScalar + GeneratedScalar + LapackScalar + PartialEq {
    fn scalar_factor() -> Self;
    fn divisor() -> Self;
    fn alpha() -> Self;
    fn beta() -> Self;
    fn denominator(row: usize, col: usize) -> Self;
    fn positive_definite(n: usize, seed: u64) -> DMatrix<Self>;
    fn lapack_real_to_f64(value: Self::Real) -> f64;
}

macro_rules! impl_real_arithmetic_scalar {
    ($ty:ty, $spd:ident) => {
        impl ArithmeticScalar for $ty {
            fn scalar_factor() -> Self {
                -1.75
            }
            fn divisor() -> Self {
                2.5
            }
            fn alpha() -> Self {
                -0.75
            }
            fn beta() -> Self {
                0.25
            }
            fn denominator(row: usize, col: usize) -> Self {
                (2 + row + 2 * col) as Self
            }
            fn positive_definite(n: usize, seed: u64) -> DMatrix<Self> {
                $spd(n, seed, 1.0)
            }
            fn lapack_real_to_f64(value: Self::Real) -> f64 {
                value as f64
            }
        }
    };
}

impl_real_arithmetic_scalar!(f32, spd_f32);
impl_real_arithmetic_scalar!(f64, spd_f64);

impl ArithmeticScalar for Complex32 {
    fn scalar_factor() -> Self {
        Self::new(-1.25, 0.75)
    }
    fn divisor() -> Self {
        Self::new(2.0, -0.5)
    }
    fn alpha() -> Self {
        Self::new(-0.75, 0.5)
    }
    fn beta() -> Self {
        Self::new(0.25, -0.125)
    }
    fn denominator(row: usize, col: usize) -> Self {
        Self::new((2 + row + 2 * col) as f32, 0.5)
    }
    fn positive_definite(n: usize, seed: u64) -> DMatrix<Self> {
        hpd_complex32(n, seed, 1.0)
    }
    fn lapack_real_to_f64(value: Self::Real) -> f64 {
        value as f64
    }
}

impl ArithmeticScalar for Complex64 {
    fn scalar_factor() -> Self {
        Self::new(-1.25, 0.75)
    }
    fn divisor() -> Self {
        Self::new(2.0, -0.5)
    }
    fn alpha() -> Self {
        Self::new(-0.75, 0.5)
    }
    fn beta() -> Self {
        Self::new(0.25, -0.125)
    }
    fn denominator(row: usize, col: usize) -> Self {
        Self::new((2 + row + 2 * col) as f64, 0.5)
    }
    fn positive_definite(n: usize, seed: u64) -> DMatrix<Self> {
        hpd_complex64(n, seed, 1.0)
    }
    fn lapack_real_to_f64(value: Self::Real) -> f64 {
        value
    }
}

fn tolerance<T: ArithmeticScalar>() -> Tolerance<f64> {
    Tolerance::for_scalar::<T>()
}

fn assert_vector_close<T: ArithmeticScalar>(expected: &DVector<T>, actual: &[T]) {
    assert_slice_close(
        expected.as_slice(),
        actual,
        tolerance::<T>(),
        expected.len(),
    );
}

fn lower_denominator<T: ArithmeticScalar>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, |row, col| {
        if row >= col {
            T::denominator(row, col)
        } else {
            T::zero()
        }
    })
}

fn upper_denominator<T: ArithmeticScalar>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, |row, col| {
        if row <= col {
            T::denominator(row, col)
        } else {
            T::zero()
        }
    })
}

fn symmetric_denominator<T: ArithmeticScalar>(n: usize) -> DMatrix<T> {
    DMatrix::from_fn(n, n, |row, col| T::denominator(row.max(col), row.min(col)))
}

macro_rules! exercise_ring_family {
    ($function:ident, $packed:ident, $generate:ident, $pack:ident, $expand:ident, $denominator:ident) => {
        fn $function<T: ArithmeticScalar>() {
            for n in DIMENSIONS {
                let full = $generate::<T>(n, 0x30_0000 + n as u64);
                let denominator = $denominator::<T>(n);
                let data = $pack(&full);
                let denominator_data = $pack(&denominator);
                let owned = $packed::from_vec(n, data.clone()).unwrap();
                let rhs = $packed::from_slice(n, &denominator_data).unwrap();

                let sum = &owned + &rhs;
                assert_matrix_close(&(&full + &denominator), &$expand(&sum), tolerance::<T>());
                let difference = &owned - &rhs;
                assert_matrix_close(
                    &(&full - &denominator),
                    &$expand(&difference),
                    tolerance::<T>(),
                );
                let negated = -&owned;
                assert_matrix_close(&(-&full), &$expand(&negated), tolerance::<T>());

                let scaled = &owned * T::scalar_factor();
                assert_matrix_close(
                    &(full.clone() * T::scalar_factor()),
                    &$expand(&scaled),
                    tolerance::<T>(),
                );
                let divided = &owned / T::divisor();
                assert_matrix_close(
                    &(full.clone() / T::divisor()),
                    &$expand(&divided),
                    tolerance::<T>(),
                );
                assert_matrix_close(
                    &DMatrix::zeros(n, n),
                    &$expand(&(&owned * T::zero())),
                    tolerance::<T>(),
                );
                assert_matrix_close(&full, &$expand(&(&owned * T::one())), tolerance::<T>());

                let component_mul = owned.component_mul(&rhs).unwrap();
                assert_matrix_close(
                    &full.component_mul(&denominator),
                    &$expand(&component_mul),
                    tolerance::<T>(),
                );
                let expected_div = DMatrix::from_fn(n, n, |row, col| {
                    if full[(row, col)] == T::zero() && denominator[(row, col)] == T::zero() {
                        T::zero()
                    } else {
                        full[(row, col)] / denominator[(row, col)]
                    }
                });
                let component_div = owned.component_div(&rhs).unwrap();
                assert_matrix_close(&expected_div, &$expand(&component_div), tolerance::<T>());

                let mut assigned_data = data.clone();
                {
                    let mut assigned = $packed::from_slice_mut(n, &mut assigned_data).unwrap();
                    assigned += &rhs;
                    assert_matrix_close(
                        &(&full + &denominator),
                        &$expand(&assigned),
                        tolerance::<T>(),
                    );
                    assigned -= &rhs;
                    assigned *= T::scalar_factor();
                    assigned /= T::divisor();
                    let expected = full.clone() * T::scalar_factor() / T::divisor();
                    assert_matrix_close(&expected, &$expand(&assigned), tolerance::<T>());
                }

                let expected_norm = data
                    .iter()
                    .map(|&value| T::magnitude(value).powi(2))
                    .sum::<f64>();
                let actual_norm = T::lapack_real_to_f64(owned.stored_norm_squared());
                let threshold = tolerance::<T>().abs + tolerance::<T>().rel * expected_norm;
                assert!((expected_norm - actual_norm).abs() <= threshold);
            }
        }
    };
}

exercise_ring_family!(
    exercise_lower_ring,
    PackedLower,
    arbitrary_lower,
    pack_lower_column_major,
    lower_to_dmatrix,
    lower_denominator
);
exercise_ring_family!(
    exercise_upper_ring,
    PackedUpper,
    arbitrary_upper,
    pack_upper_column_major,
    upper_to_dmatrix,
    upper_denominator
);

fn exercise_symmetric_ring<T: ArithmeticScalar>() {
    for n in DIMENSIONS {
        let full = symmetric_denominator::<T>(n);
        let rhs_full = DMatrix::from_fn(n, n, |row, col| {
            T::denominator(row.max(col), row.min(col)) + T::one()
        });
        let data = pack_lower_column_major(&full);
        let rhs_data = pack_lower_column_major(&rhs_full);
        let owned = PackedSymmetric::from_vec(n, data.clone()).unwrap();
        let rhs = PackedSymmetric::from_slice(n, &rhs_data).unwrap();

        assert_matrix_close(
            &(&full + &rhs_full),
            &symmetric_to_dmatrix(&(&owned + &rhs)),
            tolerance::<T>(),
        );
        assert_matrix_close(
            &(&full - &rhs_full),
            &symmetric_to_dmatrix(&(&owned - &rhs)),
            tolerance::<T>(),
        );
        assert_matrix_close(
            &(-&full),
            &symmetric_to_dmatrix(&(-&owned)),
            tolerance::<T>(),
        );
        assert_matrix_close(
            &(full.clone() * T::scalar_factor()),
            &symmetric_to_dmatrix(&(&owned * T::scalar_factor())),
            tolerance::<T>(),
        );
        assert_matrix_close(
            &(full.clone() / T::divisor()),
            &symmetric_to_dmatrix(&(&owned / T::divisor())),
            tolerance::<T>(),
        );
        assert_symmetric(&symmetric_to_dmatrix(&owned), tolerance::<T>());

        let mut backing = data;
        let mut view_mut = PackedSymmetric::from_slice_mut(n, &mut backing).unwrap();
        view_mut += &rhs;
        view_mut -= &rhs;
        view_mut *= T::scalar_factor();
        view_mut /= T::divisor();
        assert_matrix_close(
            &(full * T::scalar_factor() / T::divisor()),
            &symmetric_to_dmatrix(&view_mut),
            tolerance::<T>(),
        );
    }
}

fn exercise_spd_addition<T: ArithmeticScalar>() {
    for n in DIMENSIONS {
        let left_full = T::positive_definite(n, 0x30_1000 + n as u64);
        let right_full = T::positive_definite(n, 0x30_2000 + n as u64);
        let left_data = pack_lower_column_major(&left_full);
        let right_data = pack_lower_column_major(&right_full);
        let left = PackedSPD::from_vec(n, left_data.clone()).unwrap();
        let right = PackedSPD::from_slice(n, &right_data).unwrap();
        let sum = &left + &right;
        let expected_sum = &left_full + &right_full;
        assert_matrix_close(&expected_sum, &spd_to_dmatrix(&sum), tolerance::<T>());
        assert_hermitian(&spd_to_dmatrix(&sum), tolerance::<T>());
        assert!(expected_sum.cholesky().is_some());

        let mut backing = left_data;
        let mut view_mut = PackedSPD::from_slice_mut(n, &mut backing).unwrap();
        view_mut += &right;
        assert_matrix_close(
            &(&left_full + &right_full),
            &spd_to_dmatrix(&view_mut),
            tolerance::<T>(),
        );
    }
}

fn exercise_hermitian_ring<T: ArithmeticScalar>() {
    for n in DIMENSIONS {
        let left_full = hermitian::<T>(n, 0x30_3000 + n as u64);
        let right_full = hermitian::<T>(n, 0x30_4000 + n as u64);
        let left = PackedHermitian::from_vec(n, pack_lower_column_major(&left_full)).unwrap();
        let right_data = pack_lower_column_major(&right_full);
        let right = PackedHermitian::from_slice(n, &right_data).unwrap();
        assert_matrix_close(
            &(&left_full + &right_full),
            &hermitian_to_dmatrix(&(&left + &right)),
            tolerance::<T>(),
        );
        assert_matrix_close(
            &(&left_full - &right_full),
            &hermitian_to_dmatrix(&(&left - &right)),
            tolerance::<T>(),
        );
        let negated = -&left;
        assert_matrix_close(
            &(-&left_full),
            &hermitian_to_dmatrix(&negated),
            tolerance::<T>(),
        );
        assert_hermitian(&hermitian_to_dmatrix(&negated), tolerance::<T>());
    }
}

macro_rules! ring_scalar_tests {
    ($module:ident, $ty:ty) => {
        mod $module {
            use super::*;
            #[test]
            fn arithmetic_lower_ring_owned_view_view_mut() {
                exercise_lower_ring::<$ty>();
            }
            #[test]
            fn arithmetic_upper_ring_owned_view_view_mut() {
                exercise_upper_ring::<$ty>();
            }
            #[test]
            fn arithmetic_symmetric_ring_owned_view_view_mut() {
                exercise_symmetric_ring::<$ty>();
            }
            #[test]
            fn arithmetic_spd_addition_owned_view_view_mut() {
                exercise_spd_addition::<$ty>();
            }
            #[test]
            fn arithmetic_hermitian_add_sub_neg_owned_view() {
                exercise_hermitian_ring::<$ty>();
            }
        }
    };
}

ring_scalar_tests!(f32_arithmetic, f32);
ring_scalar_tests!(f64_arithmetic, f64);
ring_scalar_tests!(complex32_arithmetic, Complex32);
ring_scalar_tests!(complex64_arithmetic, Complex64);

fn operated_matrix<T: ArithmeticScalar>(
    full: &DMatrix<T>,
    op: Transpose,
    diagonal: Diagonal,
) -> DMatrix<T> {
    let mut logical = full.clone();
    if diagonal == Diagonal::Unit {
        for index in 0..logical.nrows() {
            logical[(index, index)] = T::one();
        }
    }
    match op {
        Transpose::None => logical,
        Transpose::Transpose => logical.transpose(),
        Transpose::ConjugateTranspose => logical.adjoint(),
    }
}

fn strided_buffer<T: ArithmeticScalar>(logical: &[T], increment: i32, padding: T) -> Vec<T> {
    if logical.is_empty() {
        return Vec::new();
    }
    let stride = increment.unsigned_abs() as usize;
    let mut buffer = vec![padding; 1 + (logical.len() - 1) * stride];
    for (index, &value) in logical.iter().enumerate() {
        let position = if increment > 0 {
            index * stride
        } else {
            (logical.len() - 1 - index) * stride
        };
        buffer[position] = value;
    }
    buffer
}

fn logical_from_strided<T: ArithmeticScalar>(buffer: &[T], n: usize, increment: i32) -> Vec<T> {
    let stride = increment.unsigned_abs() as usize;
    (0..n)
        .map(|index| {
            let position = if increment > 0 {
                index * stride
            } else {
                (n - 1 - index) * stride
            };
            buffer[position]
        })
        .collect()
}

macro_rules! triangular_vector_tests {
    ($module:ident, $ty:ty, $packed:ident, $generate:ident, $pack:ident) => {
        mod $module {
            use super::*;

            #[test]
            fn arithmetic_tpmv_owned_view_view_mut_transposes_and_unit_diagonal() {
                for n in DIMENSIONS {
                    let mut full = $generate::<$ty>(n, 0x30_5000 + n as u64);
                    for index in 0..n {
                        full[(index, index)] = <$ty as ArithmeticScalar>::denominator(index, index);
                        assert_ne!(full[(index, index)], <$ty as One>::one());
                    }
                    let data = $pack(&full);
                    let x = vector::<$ty>(n, 0x30_6000 + n as u64);
                    let owned = $packed::from_vec(n, data.clone()).unwrap();
                    let view = $packed::from_slice(n, &data).unwrap();
                    for op in [
                        Transpose::None,
                        Transpose::Transpose,
                        Transpose::ConjugateTranspose,
                    ] {
                        for diagonal in [Diagonal::NonUnit, Diagonal::Unit] {
                            let expected = operated_matrix(&full, op, diagonal) * &x;
                            let allocating =
                                owned.mul_vector_op(x.as_slice(), op, diagonal).unwrap();
                            assert_vector_close(&expected, &allocating);
                            let mut in_place = x.as_slice().to_vec();
                            view.mul_vector_op_in_place(&mut in_place, op, diagonal)
                                .unwrap();
                            assert_vector_close(&expected, &in_place);
                            let mut mutable_data = data.clone();
                            let mutable = $packed::from_slice_mut(n, &mut mutable_data).unwrap();
                            let mut through_mutable_view = x.as_slice().to_vec();
                            mutable
                                .mul_vector_op_in_place(&mut through_mutable_view, op, diagonal)
                                .unwrap();
                            assert_vector_close(&expected, &through_mutable_view);
                        }
                    }
                    assert_vector_close(&(full.clone() * &x), &(&owned * x.as_slice()));
                    assert_eq!(
                        owned.mul_vector(x.as_slice()).unwrap(),
                        owned
                            .mul_vector_op(x.as_slice(), Transpose::None, Diagonal::NonUnit)
                            .unwrap()
                    );
                }
            }

            #[test]
            fn arithmetic_tpmv_signed_strides_padding_and_validation() {
                let n = 5;
                let full = $generate::<$ty>(n, 0x30_7000);
                let matrix = $packed::from_vec(n, $pack(&full)).unwrap();
                let x = vector::<$ty>(n, 0x30_8000);
                let padding = <$ty as ArithmeticScalar>::denominator(7, 9);
                for increment in [1, 2, 3, -1, -2, -3] {
                    let mut physical = strided_buffer(x.as_slice(), increment, padding);
                    let before = physical.clone();
                    matrix
                        .mul_vector_strided_in_place(
                            &mut physical,
                            increment,
                            Transpose::None,
                            Diagonal::NonUnit,
                        )
                        .unwrap();
                    let actual = logical_from_strided(&physical, n, increment);
                    assert_vector_close(&(full.clone() * &x), &actual);
                    let stride = increment.unsigned_abs() as usize;
                    for index in 0..physical.len() {
                        if index % stride != 0 {
                            assert_eq!(physical[index], before[index]);
                        }
                    }
                }
                assert!(matches!(
                    matrix.mul_vector_strided_in_place(
                        &mut [],
                        0,
                        Transpose::None,
                        Diagonal::NonUnit
                    ),
                    Err(PackedMatrixError::InvalidIncrement { .. })
                ));
                assert!(matches!(
                    matrix.mul_vector_strided_in_place(
                        &mut [<$ty as ArithmeticScalar>::scalar_factor()],
                        2,
                        Transpose::None,
                        Diagonal::NonUnit
                    ),
                    Err(PackedMatrixError::InvalidVectorLength { .. })
                ));
            }
        }
    };
}

triangular_vector_tests!(
    lower_f32_tpmv,
    f32,
    PackedLower,
    arbitrary_lower,
    pack_lower_column_major
);
triangular_vector_tests!(
    lower_f64_tpmv,
    f64,
    PackedLower,
    arbitrary_lower,
    pack_lower_column_major
);
triangular_vector_tests!(
    lower_c32_tpmv,
    Complex32,
    PackedLower,
    arbitrary_lower,
    pack_lower_column_major
);
triangular_vector_tests!(
    lower_c64_tpmv,
    Complex64,
    PackedLower,
    arbitrary_lower,
    pack_lower_column_major
);
triangular_vector_tests!(
    upper_f32_tpmv,
    f32,
    PackedUpper,
    arbitrary_upper,
    pack_upper_column_major
);
triangular_vector_tests!(
    upper_f64_tpmv,
    f64,
    PackedUpper,
    arbitrary_upper,
    pack_upper_column_major
);
triangular_vector_tests!(
    upper_c32_tpmv,
    Complex32,
    PackedUpper,
    arbitrary_upper,
    pack_upper_column_major
);
triangular_vector_tests!(
    upper_c64_tpmv,
    Complex64,
    PackedUpper,
    arbitrary_upper,
    pack_upper_column_major
);

macro_rules! packed_vector_test {
    ($name:ident, $ty:ty, $packed:ident, $generate:expr, $expand:ident) => {
        #[test]
        fn $name() {
            for n in DIMENSIONS {
                let full: DMatrix<$ty> = ($generate)(n, 0x30_9000 + n as u64);
                let data = pack_lower_column_major(&full);
                let x = vector::<$ty>(n, 0x30_A000 + n as u64);
                let y = vector::<$ty>(n, 0x30_B000 + n as u64);
                let owned = $packed::from_vec(n, data.clone()).unwrap();
                let view = $packed::from_slice(n, &data).unwrap();
                let expected = &full * &x;
                assert_vector_close(&expected, &owned.mul_vector(x.as_slice()).unwrap());
                assert_vector_close(&expected, &(&view * x.as_slice()));

                let mut equivalent_in_place = vec![<$ty as Zero>::zero(); n];
                view.mul_vector_into(
                    x.as_slice(),
                    &mut equivalent_in_place,
                    <$ty as One>::one(),
                    <$ty as Zero>::zero(),
                )
                .unwrap();
                assert_vector_close(&expected, &equivalent_in_place);

                let mut actual = y.as_slice().to_vec();
                view.mul_vector_into(
                    x.as_slice(),
                    &mut actual,
                    <$ty as ArithmeticScalar>::alpha(),
                    <$ty as ArithmeticScalar>::beta(),
                )
                .unwrap();
                let expected_into = (&full * &x) * <$ty as ArithmeticScalar>::alpha()
                    + y.clone() * <$ty as ArithmeticScalar>::beta();
                assert_vector_close(&expected_into, &actual);

                let mut mutable_data = data.clone();
                let mutable = $packed::from_slice_mut(n, &mut mutable_data).unwrap();
                assert_vector_close(&expected, &mutable.mul_vector(x.as_slice()).unwrap());
                assert_matrix_close(&full, &$expand(&mutable), tolerance::<$ty>());
                assert_eq!(
                    owned.mul_vector(x.as_slice()).unwrap(),
                    (&owned * x.as_slice())
                );
            }
        }
    };
}

packed_vector_test!(
    arithmetic_spmv_f32_owned_view_view_mut,
    f32,
    PackedSymmetric,
    |n, seed| real_symmetric_f32(n, seed),
    symmetric_to_dmatrix
);
packed_vector_test!(
    arithmetic_spmv_f64_owned_view_view_mut,
    f64,
    PackedSymmetric,
    |n, seed| real_symmetric_f64(n, seed),
    symmetric_to_dmatrix
);
packed_vector_test!(
    arithmetic_pmv_f32_owned_view_view_mut,
    f32,
    PackedSPD,
    |n, seed| spd_f32(n, seed, 1.0),
    spd_to_dmatrix
);
packed_vector_test!(
    arithmetic_pmv_f64_owned_view_view_mut,
    f64,
    PackedSPD,
    |n, seed| spd_f64(n, seed, 1.0),
    spd_to_dmatrix
);
packed_vector_test!(
    arithmetic_pmv_c32_owned_view_view_mut,
    Complex32,
    PackedSPD,
    |n, seed| hpd_complex32(n, seed, 1.0),
    spd_to_dmatrix
);
packed_vector_test!(
    arithmetic_pmv_c64_owned_view_view_mut,
    Complex64,
    PackedSPD,
    |n, seed| hpd_complex64(n, seed, 1.0),
    spd_to_dmatrix
);
packed_vector_test!(
    arithmetic_hpmv_c32_owned_view_view_mut,
    Complex32,
    PackedHermitian,
    |n, seed| hermitian::<Complex32>(n, seed),
    hermitian_to_dmatrix
);
packed_vector_test!(
    arithmetic_hpmv_c64_owned_view_view_mut,
    Complex64,
    PackedHermitian,
    |n, seed| hermitian::<Complex64>(n, seed),
    hermitian_to_dmatrix
);

macro_rules! symmetric_rank_tests {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            for n in DIMENSIONS {
                let full =
                    DMatrix::from_fn(n, n, |r, c| (1 + r.max(c) + 2 * r.min(c)) as $ty * 0.1);
                let data = pack_lower_column_major(&full);
                let x = vector::<$ty>(n, 0x30_C000 + n as u64);
                let y = vector::<$ty>(n, 0x30_D000 + n as u64);
                let alpha: $ty = -0.625;
                let mut owned = PackedSymmetric::from_vec(n, data.clone()).unwrap();
                owned.rank1_update_in_place(alpha, x.as_slice()).unwrap();
                let expected_rank1 = &full + (&x * x.transpose()) * alpha;
                assert_matrix_close(
                    &expected_rank1,
                    &symmetric_to_dmatrix(&owned),
                    tolerance::<$ty>(),
                );
                owned
                    .rank2_update_in_place(alpha, x.as_slice(), y.as_slice())
                    .unwrap();
                let expected_rank2 =
                    expected_rank1 + ((&x * y.transpose()) + (&y * x.transpose())) * alpha;
                assert_matrix_close(
                    &expected_rank2,
                    &symmetric_to_dmatrix(&owned),
                    tolerance::<$ty>(),
                );
                assert_symmetric(&symmetric_to_dmatrix(&owned), tolerance::<$ty>());

                let padding: $ty = 91.0;
                for increment in [2, 3, -2, -3] {
                    let x_physical = strided_buffer(x.as_slice(), increment, padding);
                    let x_before = x_physical.clone();
                    let mut rank1_backing = data.clone();
                    let mut rank1_mutable =
                        PackedSymmetric::from_slice_mut(n, &mut rank1_backing).unwrap();
                    rank1_mutable
                        .rank1_update_strided_in_place(alpha, &x_physical, increment)
                        .unwrap();
                    let expected_rank1_strided = full.clone() + (&x * x.transpose()) * alpha;
                    assert_matrix_close(
                        &expected_rank1_strided,
                        &symmetric_to_dmatrix(&rank1_mutable),
                        tolerance::<$ty>(),
                    );
                    assert_eq!(x_physical, x_before);

                    let mut backing = data.clone();
                    let mut mutable = PackedSymmetric::from_slice_mut(n, &mut backing).unwrap();
                    let y_physical = strided_buffer(y.as_slice(), increment, padding);
                    let y_before = y_physical.clone();
                    mutable
                        .rank2_update_strided_in_place(
                            alpha,
                            &x_physical,
                            increment,
                            &y_physical,
                            increment,
                        )
                        .unwrap();
                    let expected =
                        full.clone() + ((&x * y.transpose()) + (&y * x.transpose())) * alpha;
                    assert_matrix_close(
                        &expected,
                        &symmetric_to_dmatrix(&mutable),
                        tolerance::<$ty>(),
                    );
                    assert_eq!(x_physical, x_before);
                    assert_eq!(y_physical, y_before);
                }
            }
        }
    };
}

symmetric_rank_tests!(arithmetic_spr_spr2_f32_owned_and_mutable_view, f32);
symmetric_rank_tests!(arithmetic_spr_spr2_f64_owned_and_mutable_view, f64);

macro_rules! hermitian_rank_tests {
    ($name:ident, $ty:ty, $real:ty) => {
        #[test]
        fn $name() {
            for n in DIMENSIONS {
                let full = hermitian::<$ty>(n, 0x30_E000 + n as u64);
                let data = pack_lower_column_major(&full);
                let x = vector::<$ty>(n, 0x30_F000 + n as u64);
                let y = vector::<$ty>(n, 0x31_0000 + n as u64);
                let alpha_real: $real = -0.375;
                let alpha = <$ty as ArithmeticScalar>::alpha();
                let mut owned = PackedHermitian::from_vec(n, data.clone()).unwrap();
                owned
                    .rank1_update_in_place(alpha_real, x.as_slice())
                    .unwrap();
                let expected_rank1 = &full + (&x * x.adjoint()) * <$ty>::new(alpha_real, 0.0);
                assert_matrix_close(
                    &expected_rank1,
                    &hermitian_to_dmatrix(&owned),
                    tolerance::<$ty>(),
                );
                owned
                    .rank2_update_in_place(alpha, x.as_slice(), y.as_slice())
                    .unwrap();
                let expected_rank2 =
                    expected_rank1 + (&x * y.adjoint()) * alpha + (&y * x.adjoint()) * alpha.conj();
                assert_matrix_close(
                    &expected_rank2,
                    &hermitian_to_dmatrix(&owned),
                    tolerance::<$ty>(),
                );
                assert_hermitian(&hermitian_to_dmatrix(&owned), tolerance::<$ty>());
                for index in 0..n {
                    assert_eq!(owned.get(index, index).unwrap().im, 0.0);
                }

                let padding = <$ty>::new(91.0, -17.0);
                for increment in [2, 3, -2, -3] {
                    let x_physical = strided_buffer(x.as_slice(), increment, padding);
                    let x_before = x_physical.clone();
                    let mut rank1_backing = data.clone();
                    let mut rank1_mutable =
                        PackedHermitian::from_slice_mut(n, &mut rank1_backing).unwrap();
                    rank1_mutable
                        .rank1_update_strided_in_place(alpha_real, &x_physical, increment)
                        .unwrap();
                    let expected_rank1_strided =
                        full.clone() + (&x * x.adjoint()) * <$ty>::new(alpha_real, 0.0);
                    assert_matrix_close(
                        &expected_rank1_strided,
                        &hermitian_to_dmatrix(&rank1_mutable),
                        tolerance::<$ty>(),
                    );
                    assert_hermitian(&hermitian_to_dmatrix(&rank1_mutable), tolerance::<$ty>());
                    assert_eq!(x_physical, x_before);

                    let mut backing = data.clone();
                    let mut mutable = PackedHermitian::from_slice_mut(n, &mut backing).unwrap();
                    let y_physical = strided_buffer(y.as_slice(), increment, padding);
                    let y_before = y_physical.clone();
                    mutable
                        .rank2_update_strided_in_place(
                            alpha,
                            &x_physical,
                            increment,
                            &y_physical,
                            increment,
                        )
                        .unwrap();
                    let expected = full.clone()
                        + (&x * y.adjoint()) * alpha
                        + (&y * x.adjoint()) * alpha.conj();
                    assert_matrix_close(
                        &expected,
                        &hermitian_to_dmatrix(&mutable),
                        tolerance::<$ty>(),
                    );
                    assert_hermitian(&hermitian_to_dmatrix(&mutable), tolerance::<$ty>());
                    assert_eq!(x_physical, x_before);
                    assert_eq!(y_physical, y_before);
                }
            }
        }
    };
}

hermitian_rank_tests!(
    arithmetic_hpr_hpr2_c32_owned_and_mutable_view,
    Complex32,
    f32
);
hermitian_rank_tests!(
    arithmetic_hpr_hpr2_c64_owned_and_mutable_view,
    Complex64,
    f64
);

#[test]
fn arithmetic_dimension_and_vector_errors_are_reported() {
    let lower2 = PackedLower::from_vec(2, vec![1.0f64, 2.0, 3.0]).unwrap();
    let lower3 = PackedLower::from_vec(3, vec![1.0f64; 6]).unwrap();
    assert!(std::panic::catch_unwind(|| &lower2 + &lower3).is_err());
    assert!(matches!(
        lower2.component_mul(&lower3),
        Err(PackedMatrixError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        lower2.component_div(&lower3),
        Err(PackedMatrixError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        lower2.mul_vector(&[1.0]),
        Err(PackedMatrixError::InvalidVectorLength { .. })
    ));

    let mut symmetric = PackedSymmetric::from_vec(2, vec![1.0f64, 0.0, 1.0]).unwrap();
    let mut short_y = [0.0];
    assert!(matches!(
        symmetric.mul_vector_into(&[1.0, 2.0], &mut short_y, 1.0, 0.0),
        Err(PackedMatrixError::InvalidVectorLength { .. })
    ));
    assert!(matches!(
        symmetric.rank1_update_strided_in_place(1.0, &[1.0, 2.0], 0),
        Err(PackedMatrixError::InvalidIncrement { .. })
    ));
    assert!(matches!(
        symmetric.rank2_update_strided_in_place(1.0, &[1.0], 2, &[1.0, 2.0], 1),
        Err(PackedMatrixError::InvalidVectorLength { .. })
    ));

    let symmetric3 = PackedSymmetric::from_vec(3, vec![1.0f64; 6]).unwrap();
    let spd2 = PackedSPD::from_vec(2, vec![2.0f64, 0.0, 2.0]).unwrap();
    let spd3 = PackedSPD::from_vec(3, vec![3.0f64, 0.0, 0.0, 3.0, 0.0, 3.0]).unwrap();
    let hermitian2 = PackedHermitian::from_vec(2, vec![1.0f64, 0.0, 1.0]).unwrap();
    let hermitian3 = PackedHermitian::from_vec(3, vec![1.0f64; 6]).unwrap();
    assert!(std::panic::catch_unwind(|| &symmetric + &symmetric3).is_err());
    assert!(std::panic::catch_unwind(|| &spd2 + &spd3).is_err());
    assert!(std::panic::catch_unwind(|| &hermitian2 + &hermitian3).is_err());

    let c = Complex64::new;
    let mut hermitian_rank =
        PackedHermitian::from_vec(2, vec![c(1.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)]).unwrap();
    assert!(matches!(
        hermitian_rank.rank1_update_strided_in_place(1.0, &[c(1.0, 0.0)], 0),
        Err(PackedMatrixError::InvalidIncrement { .. })
    ));
    assert!(matches!(
        hermitian_rank.rank2_update_strided_in_place(
            c(1.0, 0.0),
            &[c(1.0, 0.0)],
            2,
            &[c(1.0, 0.0), c(2.0, 0.0)],
            1,
        ),
        Err(PackedMatrixError::InvalidVectorLength { .. })
    ));
}

proptest! {
    #[test]
    fn arithmetic_property_lower_scale_add_and_tpmv_match_nalgebra(
        n in 0usize..12,
        scale in -2.0f64..2.0,
    ) {
        let full = arbitrary_lower::<f64>(n, 0x31_1000 + n as u64);
        let rhs_full = arbitrary_lower::<f64>(n, 0x31_2000 + n as u64);
        let matrix = PackedLower::from_vec(n, pack_lower_column_major(&full)).unwrap();
        let rhs = PackedLower::from_vec(n, pack_lower_column_major(&rhs_full)).unwrap();
        assert_matrix_close(&(full.clone() * scale), &lower_to_dmatrix(&(&matrix * scale)), tolerance::<f64>());
        assert_matrix_close(&(&full + &rhs_full), &lower_to_dmatrix(&(&matrix + &rhs)), tolerance::<f64>());
        let x = vector::<f64>(n, 0x31_3000 + n as u64);
        assert_vector_close(&(full * &x), &matrix.mul_vector(x.as_slice()).unwrap());
    }
}
