//! Demonstrates DLATPS, which solves a packed triangular system with overflow-protecting LAPACK scaling.
//! The matrix remains in standard lower-triangular packed-column storage; the
//! operation passes that packed slice directly to LAPACK without expanding it to a dense matrix.

use matrixpacked::{PackedLower, Diagonal, Transpose};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedLower::<f64>::from_vec(2, vec![2f64, 1f64, 3f64])?;
    let mut b = [2f64, 7f64];
    let scale = a.solve_scaled_in_place(&mut b, Transpose::None, Diagonal::NonUnit)?;
    assert!(scale > 0.0);
    assert!((b[0] - 1f64).abs() < 1e-10);
    assert!((b[1] - 2f64).abs() < 1e-10);
    Ok(())
}
