use matrixpacked::{MatrixNorm, PackedSymmetric};

fn approx_eq(actual: f32, expected: f32) {
    assert!((actual - expected).abs() <= 1.0e-5, "actual={actual}, expected={expected}");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // [1 2]
    // [2 3]
    let a = PackedSymmetric::<f32>::from_vec(2, vec![1.0, 2.0, 3.0])?;

    approx_eq(a.matrix_norm(MatrixNorm::MaxAbs)?, 3.0);
    approx_eq(a.matrix_norm(MatrixNorm::One)?, 5.0);
    approx_eq(a.matrix_norm(MatrixNorm::Infinity)?, 5.0);
    approx_eq(a.matrix_norm(MatrixNorm::Frobenius)?, 18.0_f32.sqrt());
    Ok(())
}
