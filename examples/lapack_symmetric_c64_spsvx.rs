//! Demonstrates ZSPSVX expert complex-symmetric packed solving.
use matrixpacked::PackedSymmetric;
use num_complex::Complex64;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex64::new;
    let a = PackedSymmetric::from_vec(2, vec![c(1e-8, 0.), c(1., 1.), c(-1e8, 0.)])?;
    let r = a.expert_solve(&[c(1., 0.), c(1., 0.)], 1)?;
    assert_eq!(r.forward_error.len(), 1);
    assert!(r.reciprocal_condition_number >= 0.);
    Ok(())
}
