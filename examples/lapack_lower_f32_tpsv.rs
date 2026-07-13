//! Demonstrates STPSV, which solves one packed triangular system in place with the Level-2 BLAS routine.
//! The matrix remains in standard lower-triangular packed-column storage; the
//! operation passes that packed slice directly to BLAS without expanding it to a dense matrix.

use matrixpacked::{Diagonal, PackedLower, Transpose};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedLower::<f32>::from_vec(2, vec![2f32, 1f32, 3f32])?;
    let mut b = [2f32, 7f32];
    a.solve_vector_blas_in_place(&mut b, Transpose::None, Diagonal::NonUnit)?;
    assert!((b[0] - 1f32).abs() < 1e-4);
    assert!((b[1] - 2f32).abs() < 1e-4);
    Ok(())
}
