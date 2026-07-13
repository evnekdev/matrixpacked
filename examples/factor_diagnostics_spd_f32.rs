//! Derives a stable log determinant from a packed Cholesky factor.

use matrixpacked::PackedSPD;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let factor = PackedSPD::from_vec(2, vec![4.0f32, 0.0, 9.0])?.cholesky()?;
    assert!((factor.log_determinant() - 36.0f32.ln()).abs() < 1e-5);
    assert!((factor.determinant() - 36.0).abs() < 1e-4);
    Ok(())
}
