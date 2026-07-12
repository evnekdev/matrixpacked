use matrixpacked::{MatrixNorm, PackedSPD};

fn approx_eq(actual: f32, expected: f32) {
    assert!((actual - expected).abs() <= 1.0e-5, "actual={actual}, expected={expected}");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // [4 1]
    // [1 3]
    let a = PackedSPD::<f32>::from_vec(2, vec![4.0, 1.0, 3.0])?;

    approx_eq(a.matrix_norm(MatrixNorm::MaxAbs)?, 4.0);
    approx_eq(a.matrix_norm(MatrixNorm::One)?, 5.0);
    approx_eq(a.matrix_norm(MatrixNorm::Infinity)?, 5.0);
    approx_eq(a.matrix_norm(MatrixNorm::Frobenius)?, 27.0_f32.sqrt());
    Ok(())
}
