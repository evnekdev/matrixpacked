//! Demonstrates CHPSVX expert Hermitian packed solving.
use matrixpacked::PackedHermitian;
use num_complex::Complex32;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex32::new;
    let a = PackedHermitian::from_vec(2, vec![c(0., 0.), c(1., -1.), c(0., 0.)])?;
    let r = a.expert_solve(&[c(2., 0.), c(0., 1.)], 1)?;
    assert_eq!(r.solution.len(), 2);
    assert_eq!(r.forward_error.len(), 1);
    assert!(r.reciprocal_condition_number > 0.);
    Ok(())
}
