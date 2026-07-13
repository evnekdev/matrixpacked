//! Demonstrates CPPEQU for a complex Hermitian positive-definite packed matrix.

use matrixpacked::PackedSPD;
use num_complex::Complex32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex32::new;
    let mut a = PackedSPD::from_vec(2, vec![c(4.0, 3.0), c(1.0, -1.0), c(16.0, -2.0)])?;
    let factors = a.equilibrate_in_place()?;

    assert_eq!(factors.scaling.len(), 2);
    assert!(factors.scaling.iter().all(|s| s.is_finite() && *s > 0.0));
    assert!((factors.condition_ratio - 0.5).abs() < 1e-6);
    assert_eq!(factors.maximum_diagonal, 16.0);
    for i in 0..2 {
        let diagonal = a.get(i, i)?;
        assert!((diagonal.re - 1.0).abs() < 1e-6);
        assert_eq!(diagonal.im, 0.0);
    }
    assert_eq!(a.get(0, 1)?, a.get(1, 0)?.conj());
    Ok(())
}
