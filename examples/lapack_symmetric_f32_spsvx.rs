//! Demonstrates SSPSVX expert symmetric packed solving.
use matrixpacked::PackedSymmetric;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::from_vec(2, vec![0f32, 1., 0.])?;
    let r = a.expert_solve(&[2., 3.], 1)?;
    assert_eq!(r.solution.len(), 2);
    assert!(r.reciprocal_condition_number > 0.);
    assert!((r.solution[0] - 3.).abs() < 1e-5 && (r.solution[1] - 2.).abs() < 1e-5);
    Ok(())
}
