//! Demonstrates SPPSVX expert packed solving.
use matrixpacked::PackedSPD;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSPD::from_vec(2, vec![4f32, 1., 3.])?;
    let r = a.expert_solve(&[6., 7.], 1)?;
    assert_eq!(r.solution.len(), 2);
    assert_eq!(r.forward_error.len(), 1);
    assert_eq!(r.backward_error.len(), 1);
    assert!(r.reciprocal_condition_number > 0.);
    assert!((4. * r.solution[0] + r.solution[1] - 6.).abs() < 1e-4);
    Ok(())
}
