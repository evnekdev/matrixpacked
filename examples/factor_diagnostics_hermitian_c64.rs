//! Derives real determinant and inertia diagnostics from a Hermitian factor.

use matrixpacked::{Inertia, PackedHermitian};
use num_complex::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex64::new;
    let factor =
        PackedHermitian::from_vec(2, vec![c(0.0, 0.0), c(0.0, 1.0), c(0.0, 0.0)])?.factorize()?;
    assert_eq!(
        factor.inertia(),
        Inertia {
            positive: 1,
            negative: 1,
            zero: 0
        }
    );
    assert_eq!(factor.slogdet().sign, -1.0);
    Ok(())
}
