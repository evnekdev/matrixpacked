//! Demonstrates CPPSVX expert HPD packed solving.
use matrixpacked::PackedSPD;
use num_complex::Complex32;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex32::new;
    let a = PackedSPD::from_vec(2, vec![c(4., 0.), c(1., -1.), c(3., 0.)])?;
    let b = [c(2., 1.), c(1., -2.)];
    let r = a.expert_solve(&b, 1)?;
    assert_eq!(r.solution.len(), 2);
    assert_eq!(r.forward_error.len(), 1);
    assert!(r.reciprocal_condition_number > 0.);
    Ok(())
}
