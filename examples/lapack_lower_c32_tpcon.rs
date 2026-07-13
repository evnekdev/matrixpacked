//! Demonstrates CTPCON, which estimates the reciprocal condition number of a packed triangular matrix.
//! The matrix remains in standard lower-triangular packed-column storage; the
//! operation passes that packed slice directly to LAPACK without expanding it to a dense matrix.

use matrixpacked::{ConditionNorm, Diagonal, PackedLower};
use num_complex::Complex32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedLower::<Complex32>::from_vec(
        2,
        vec![
            Complex32::new(2.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(3.0, 0.0),
        ],
    )?;
    let rcond = a.reciprocal_condition_number(ConditionNorm::One, Diagonal::NonUnit)?;
    assert!(rcond > 0.0 && rcond <= 1.0);
    Ok(())
}
