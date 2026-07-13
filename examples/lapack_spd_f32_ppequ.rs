//! Demonstrates SPPEQU, the LAPACK packed positive-definite equilibration routine.

use matrixpacked::PackedSPD;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = PackedSPD::from_vec(2, vec![4.0f32, 1.0, 16.0])?;
    let factors = a.equilibrate_in_place()?;

    assert_eq!(factors.scaling.len(), 2);
    assert!(factors.scaling.iter().all(|s| s.is_finite() && *s > 0.0));
    assert!((factors.condition_ratio - 0.5).abs() < 1e-6);
    assert_eq!(factors.maximum_diagonal, 16.0);
    assert!((a.get(0, 0)? - 1.0).abs() < 1e-6);
    assert!((a.get(1, 1)? - 1.0).abs() < 1e-6);
    Ok(())
}
