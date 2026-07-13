//! Demonstrates CTPSV, which solves one packed triangular system in place with the Level-2 BLAS routine.
//! The matrix remains in standard upper-triangular packed-column storage; the
//! operation passes that packed slice directly to BLAS without expanding it to a dense matrix.

use matrixpacked::{Diagonal, PackedUpper, Transpose};
use num_complex::Complex32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedUpper::<Complex32>::from_vec(
        2,
        vec![
            Complex32::new(2.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(3.0, 0.0),
        ],
    )?;
    let mut b = [Complex32::new(4.0, 0.0), Complex32::new(6.0, 0.0)];
    a.solve_vector_blas_in_place(&mut b, Transpose::None, Diagonal::NonUnit)?;
    assert!((b[0] - Complex32::new(1.0, 0.0)).norm() < 1e-4);
    assert!((b[1] - Complex32::new(2.0, 0.0)).norm() < 1e-4);
    Ok(())
}
