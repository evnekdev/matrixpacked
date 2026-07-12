//! Demonstrates DTPSV, which solves one packed triangular system in place with the Level-2 BLAS routine.
//! The matrix remains in standard upper-triangular packed-column storage; the
//! operation passes that packed slice directly to BLAS without expanding it to a dense matrix.

use matrixpacked::{PackedUpper, Diagonal, Transpose};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedUpper::<f64>::from_vec(2, vec![2f64, 1f64, 3f64])?;
    let mut b = [4f64, 6f64];
    a.solve_vector_blas_in_place(&mut b, Transpose::None, Diagonal::NonUnit)?;
    assert!((b[0] - 1f64).abs() < 1e-10);
    assert!((b[1] - 2f64).abs() < 1e-10);
    Ok(())
}
