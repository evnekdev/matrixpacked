use matrixpacked::{GeneralizedEigenproblem, PackedHermitian, PackedSPD};
use num_complex::Complex64;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex64::new;
    let a = PackedHermitian::from_vec(2, vec![c(4., 0.), c(0., 0.), c(9., 0.)])?;
    let b = PackedSPD::from_vec(2, vec![c(2., 0.), c(0., 0.), c(3., 0.)])?;
    let e = a.generalized_eigendecomposition_divide_conquer(
        &b,
        GeneralizedEigenproblem::AxEqualsLambdaBx,
    )?;
    assert!((e.eigenvalues[0] - 2.).abs() < 1e-10 && (e.eigenvalues[1] - 3.).abs() < 1e-10);
    Ok(())
}
