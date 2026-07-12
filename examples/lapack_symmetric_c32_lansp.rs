use matrixpacked::{MatrixNorm, PackedSymmetric};
use num_complex::Complex32;

fn approx_eq(actual: f32, expected: f32) {
    assert!((actual - expected).abs() <= 1.0e-5, "actual={actual}, expected={expected}");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Complex symmetric, not Hermitian:
    // [1      1+2i]
    // [1+2i   3i  ]
    let a = PackedSymmetric::<Complex32>::from_vec(
        2,
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(1.0, 2.0),
            Complex32::new(0.0, 3.0),
        ],
    )?;

    approx_eq(a.matrix_norm(MatrixNorm::MaxAbs)?, 3.0);
    approx_eq(a.matrix_norm(MatrixNorm::One)?, 3.0 + 5.0_f32.sqrt());
    approx_eq(a.matrix_norm(MatrixNorm::Infinity)?, 3.0 + 5.0_f32.sqrt());
    approx_eq(a.matrix_norm(MatrixNorm::Frobenius)?, 20.0_f32.sqrt());
    Ok(())
}
