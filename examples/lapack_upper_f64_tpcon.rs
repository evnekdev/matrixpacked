//! Demonstrates DTPCON, which estimates the reciprocal condition number of a packed triangular matrix.
//! The matrix remains in standard upper-triangular packed-column storage; the
//! operation passes that packed slice directly to LAPACK without expanding it to a dense matrix.

use matrixpacked::{PackedUpper, Diagonal, ConditionNorm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedUpper::<f64>::from_vec(2, vec![2f64, 1f64, 3f64])?;
    let rcond = a.reciprocal_condition_number(ConditionNorm::One, Diagonal::NonUnit)?;
    assert!(rcond > 0.0 && rcond <= 1.0);
    Ok(())
}
