//! Demonstrates ZPPEQU for a complex Hermitian positive-definite packed matrix.

use matrixpacked::PackedSPD;
use num_complex::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex64::new;
    let mut a = PackedSPD::from_vec(2, vec![c(9.0, 4.0), c(2.0, -1.0), c(36.0, -7.0)])?;
    let factors = a.equilibrate_in_place()?;

    assert_eq!(factors.scaling.len(), 2);
    assert!(factors.scaling.iter().all(|s| s.is_finite() && *s > 0.0));
    assert!((factors.condition_ratio - 0.5).abs() < 1e-12);
    assert_eq!(factors.maximum_diagonal, 36.0);
    for i in 0..2 {
        let diagonal = a.get(i, i)?;
        assert!((diagonal.re - 1.0).abs() < 1e-12);
        assert_eq!(diagonal.im, 0.0);
    }
    assert_eq!(a.get(0, 1)?, a.get(1, 0)?.conj());
    Ok(())
}
