use matrixpacked::{MatrixNorm, PackedSPD};
use num_complex::Complex32;

fn approx_eq(actual: f32, expected: f32) {
    assert!((actual - expected).abs() <= 1.0e-5, "actual={actual}, expected={expected}");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Hermitian positive definite:
    // [4      1+i]
    // [1-i    3  ]
    let a = PackedSPD::<Complex32>::from_vec(
        2,
        vec![
            Complex32::new(4.0, 0.0),
            Complex32::new(1.0, -1.0),
            Complex32::new(3.0, 0.0),
        ],
    )?;

    approx_eq(a.matrix_norm(MatrixNorm::MaxAbs)?, 4.0);
    approx_eq(a.matrix_norm(MatrixNorm::One)?, 4.0 + 2.0_f32.sqrt());
    approx_eq(a.matrix_norm(MatrixNorm::Infinity)?, 4.0 + 2.0_f32.sqrt());
    approx_eq(a.matrix_norm(MatrixNorm::Frobenius)?, 29.0_f32.sqrt());
    Ok(())
}
