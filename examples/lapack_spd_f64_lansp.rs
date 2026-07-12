use matrixpacked::{MatrixNorm, PackedSPD};

fn approx_eq(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= 1.0e-12, "actual={actual}, expected={expected}");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSPD::<f64>::from_vec(2, vec![4.0, 1.0, 3.0])?;

    approx_eq(a.matrix_norm(MatrixNorm::MaxAbs)?, 4.0);
    approx_eq(a.matrix_norm(MatrixNorm::One)?, 5.0);
    approx_eq(a.matrix_norm(MatrixNorm::Infinity)?, 5.0);
    approx_eq(a.matrix_norm(MatrixNorm::Frobenius)?, 27.0_f64.sqrt());
    Ok(())
}
