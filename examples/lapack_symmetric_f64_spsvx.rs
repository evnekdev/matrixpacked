//! Demonstrates DSPSVX expert symmetric packed solving.
use matrixpacked::PackedSymmetric;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::from_vec(2, vec![1e-8f64, 1., -1e8])?;
    let b = [1.00000001, -99999999.];
    let r = a.expert_solve(&b, 1)?;
    assert_eq!(r.forward_error.len(), 1);
    assert_eq!(r.backward_error.len(), 1);
    assert!(r.reciprocal_condition_number >= 0.);
    Ok(())
}
