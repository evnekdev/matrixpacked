//! Demonstrates STPCON, which estimates the reciprocal condition number of a packed triangular matrix.
//! The matrix remains in standard upper-triangular packed-column storage; the
//! operation passes that packed slice directly to LAPACK without expanding it to a dense matrix.

use matrixpacked::{ConditionNorm, Diagonal, PackedUpper};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedUpper::<f32>::from_vec(2, vec![2f32, 1f32, 3f32])?;
    let rcond = a.reciprocal_condition_number(ConditionNorm::One, Diagonal::NonUnit)?;
    assert!(rcond > 0.0 && rcond <= 1.0);
    Ok(())
}
