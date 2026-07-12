//! Demonstrates STPSV, which solves one packed triangular system in place with the Level-2 BLAS routine.
//! The matrix remains in standard upper-triangular packed-column storage; the
//! operation passes that packed slice directly to BLAS without expanding it to a dense matrix.

use matrixpacked::{PackedUpper, Diagonal, Transpose};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedUpper::<f32>::from_vec(2, vec![2f32, 1f32, 3f32])?;
    let mut b = [4f32, 6f32];
    a.solve_vector_blas_in_place(&mut b, Transpose::None, Diagonal::NonUnit)?;
    assert!((b[0] - 1f32).abs() < 1e-4);
    assert!((b[1] - 2f32).abs() < 1e-4);
    Ok(())
}
