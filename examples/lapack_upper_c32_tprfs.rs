//! Demonstrates CTPRFS, which improves packed triangular solutions and reports forward/backward error estimates.
//! The matrix remains in standard upper-triangular packed-column storage; the
//! operation passes that packed slice directly to LAPACK without expanding it to a dense matrix.

use matrixpacked::{PackedUpper, Diagonal, Transpose};
use num_complex::Complex32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedUpper::<Complex32>::from_vec(2, vec![Complex32::new(2.0, 0.0), Complex32::new(1.0, 0.0), Complex32::new(3.0, 0.0)])?;
    let b = [Complex32::new(4.0, 0.0), Complex32::new(6.0, 0.0)];
    let mut x = [Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)];
    let report = a.refine_many_in_place(&b, &mut x, 1, Transpose::None, Diagonal::NonUnit)?;
    assert_eq!(report.forward_error.len(), 1);
    assert_eq!(report.backward_error.len(), 1);
    assert!((x[0] - Complex32::new(1.0, 0.0)).norm() < 1e-4);
    assert!((x[1] - Complex32::new(2.0, 0.0)).norm() < 1e-4);
    Ok(())
}
