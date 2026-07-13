//! Computes a complex triangular determinant and its real log magnitude.

use matrixpacked::{Diagonal, PackedUpper};
use num_complex::Complex32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex32::new;
    let matrix = PackedUpper::from_vec(2, vec![c(1.0, 1.0), c(4.0, 0.0), c(0.0, 2.0)])?;
    assert_eq!(matrix.determinant(Diagonal::NonUnit), c(-2.0, 2.0));
    assert!((matrix.log_abs_determinant(Diagonal::NonUnit) - 8.0f32.ln() / 2.0).abs() < 1e-6);
    assert_eq!(matrix.determinant(Diagonal::Unit), c(1.0, 0.0));
    Ok(())
}
