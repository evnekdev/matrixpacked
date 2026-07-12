//! Demonstrates ZTPSV, which solves one packed triangular system in place with the Level-2 BLAS routine.
//! The matrix remains in standard upper-triangular packed-column storage; the
//! operation passes that packed slice directly to BLAS without expanding it to a dense matrix.

use matrixpacked::{PackedUpper, Diagonal, Transpose};
use num_complex::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedUpper::<Complex64>::from_vec(2, vec![Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)])?;
    let mut b = [Complex64::new(4.0, 0.0), Complex64::new(6.0, 0.0)];
    a.solve_vector_blas_in_place(&mut b, Transpose::None, Diagonal::NonUnit)?;
    assert!((b[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    assert!((b[1] - Complex64::new(2.0, 0.0)).norm() < 1e-10);
    Ok(())
}
