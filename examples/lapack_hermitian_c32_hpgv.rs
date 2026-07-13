use matrixpacked::{GeneralizedEigenproblem, PackedHermitian, PackedSPD};
use num_complex::Complex32;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex32::new;
    let a = PackedHermitian::from_vec(2, vec![c(4., 0.), c(0., 0.), c(9., 0.)])?;
    let b = PackedSPD::from_vec(2, vec![c(2., 0.), c(0., 0.), c(3., 0.)])?;
    let v = a.generalized_eigenvalues(&b, GeneralizedEigenproblem::AxEqualsLambdaBx)?;
    assert!((v[0] - 2.).abs() < 1e-4 && (v[1] - 3.).abs() < 1e-4);
    Ok(())
}
