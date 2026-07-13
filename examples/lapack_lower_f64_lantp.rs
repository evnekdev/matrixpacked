//! Demonstrates DLANTP, which computes a selected norm of a packed triangular matrix.
//! The matrix remains in standard lower-triangular packed-column storage; the
//! operation passes that packed slice directly to LAPACK without expanding it to a dense matrix.

use matrixpacked::{Diagonal, MatrixNorm, PackedLower};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedLower::<f64>::from_vec(2, vec![2f64, 1f64, 3f64])?;
    let norm = a.matrix_norm(MatrixNorm::One, Diagonal::NonUnit)?;
    assert!((norm - 3.0).abs() < 1e-10);
    Ok(())
}
