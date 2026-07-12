//! Demonstrates DTPRFS, which improves packed triangular solutions and reports forward/backward error estimates.
//! The matrix remains in standard upper-triangular packed-column storage; the
//! operation passes that packed slice directly to LAPACK without expanding it to a dense matrix.

use matrixpacked::{PackedUpper, Diagonal, Transpose};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedUpper::<f64>::from_vec(2, vec![2f64, 1f64, 3f64])?;
    let b = [4f64, 6f64];
    let mut x = [1f64, 2f64];
    let report = a.refine_many_in_place(&b, &mut x, 1, Transpose::None, Diagonal::NonUnit)?;
    assert_eq!(report.forward_error.len(), 1);
    assert_eq!(report.backward_error.len(), 1);
    assert!((x[0] - 1f64).abs() < 1e-10);
    assert!((x[1] - 2f64).abs() < 1e-10);
    Ok(())
}
