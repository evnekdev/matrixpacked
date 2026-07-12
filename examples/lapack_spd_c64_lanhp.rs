use matrixpacked::{MatrixNorm, PackedSPD};
use num_complex::Complex64;

fn approx_eq(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= 1.0e-12, "actual={actual}, expected={expected}");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSPD::<Complex64>::from_vec(
        2,
        vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(3.0, 0.0),
        ],
    )?;

    approx_eq(a.matrix_norm(MatrixNorm::MaxAbs)?, 4.0);
    approx_eq(a.matrix_norm(MatrixNorm::One)?, 4.0 + 2.0_f64.sqrt());
    approx_eq(a.matrix_norm(MatrixNorm::Infinity)?, 4.0 + 2.0_f64.sqrt());
    approx_eq(a.matrix_norm(MatrixNorm::Frobenius)?, 29.0_f64.sqrt());
    Ok(())
}
