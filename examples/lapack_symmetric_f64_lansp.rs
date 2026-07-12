use matrixpacked::{MatrixNorm, PackedSymmetric};

fn approx_eq(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= 1.0e-12, "actual={actual}, expected={expected}");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::<f64>::from_vec(2, vec![1.0, 2.0, 3.0])?;

    approx_eq(a.matrix_norm(MatrixNorm::MaxAbs)?, 3.0);
    approx_eq(a.matrix_norm(MatrixNorm::One)?, 5.0);
    approx_eq(a.matrix_norm(MatrixNorm::Infinity)?, 5.0);
    approx_eq(a.matrix_norm(MatrixNorm::Frobenius)?, 18.0_f64.sqrt());
    Ok(())
}
