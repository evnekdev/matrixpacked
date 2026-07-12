//! Demonstrates STPRFS, which improves packed triangular solutions and reports forward/backward error estimates.
//! The matrix remains in standard lower-triangular packed-column storage; the
//! operation passes that packed slice directly to LAPACK without expanding it to a dense matrix.

use matrixpacked::{PackedLower, Diagonal, Transpose};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedLower::<f32>::from_vec(2, vec![2f32, 1f32, 3f32])?;
    let b = [2f32, 7f32];
    let mut x = [1f32, 2f32];
    let report = a.refine_many_in_place(&b, &mut x, 1, Transpose::None, Diagonal::NonUnit)?;
    assert_eq!(report.forward_error.len(), 1);
    assert_eq!(report.backward_error.len(), 1);
    assert!((x[0] - 1f32).abs() < 1e-4);
    assert!((x[1] - 2f32).abs() < 1e-4);
    Ok(())
}
