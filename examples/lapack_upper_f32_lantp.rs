//! Demonstrates SLANTP, which computes a selected norm of a packed triangular matrix.
//! The matrix remains in standard upper-triangular packed-column storage; the
//! operation passes that packed slice directly to LAPACK without expanding it to a dense matrix.

use matrixpacked::{Diagonal, MatrixNorm, PackedUpper};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedUpper::<f32>::from_vec(2, vec![2f32, 1f32, 3f32])?;
    let norm = a.matrix_norm(MatrixNorm::One, Diagonal::NonUnit)?;
    assert!((norm - 4.0).abs() < 1e-4);
    Ok(())
}
