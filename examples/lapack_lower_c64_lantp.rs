//! Demonstrates ZLANTP, which computes a selected norm of a packed triangular matrix.
//! The matrix remains in standard lower-triangular packed-column storage; the
//! operation passes that packed slice directly to LAPACK without expanding it to a dense matrix.

use matrixpacked::{Diagonal, MatrixNorm, PackedLower};
use num_complex::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedLower::<Complex64>::from_vec(
        2,
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 0.0),
        ],
    )?;
    let norm = a.matrix_norm(MatrixNorm::One, Diagonal::NonUnit)?;
    assert!((norm - 3.0).abs() < 1e-10);
    Ok(())
}
