//! Demonstrates DPPEQU, the LAPACK packed positive-definite equilibration routine.

use matrixpacked::PackedSPD;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = PackedSPD::from_vec(2, vec![1.0e-8f64, 0.25, 1.0e8])?;
    let factors = a.equilibrate_in_place()?;

    assert_eq!(factors.scaling.len(), 2);
    assert!(factors.scaling.iter().all(|s| s.is_finite() && *s > 0.0));
    assert!((factors.condition_ratio - 1.0e-8).abs() < 1e-20);
    assert_eq!(factors.maximum_diagonal, 1.0e8);
    assert!((a.get(0, 0)? - 1.0).abs() < 1e-12);
    assert!((a.get(1, 1)? - 1.0).abs() < 1e-12);
    Ok(())
}
