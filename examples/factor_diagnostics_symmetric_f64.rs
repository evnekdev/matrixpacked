//! Derives determinant sign, log magnitude, and inertia from a packed LDLT factor.

use matrixpacked::{Inertia, PackedSymmetric};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let factor = PackedSymmetric::from_vec(2, vec![0.0f64, 1.0, 0.0])?.factorize()?;
    assert_eq!(
        factor.inertia(),
        Inertia {
            positive: 1,
            negative: 1,
            zero: 0
        }
    );
    assert_eq!(factor.slogdet().sign, -1.0);
    assert_eq!(factor.slogdet().log_abs, 0.0);
    Ok(())
}
