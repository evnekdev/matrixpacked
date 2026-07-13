use matrixpacked::{EigenRange, GeneralizedEigenproblem, PackedHermitian, PackedSPD};
use num_complex::Complex64;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex64::new;
    let a = PackedHermitian::from_vec(2, vec![c(4., 0.), c(0., 0.), c(9., 0.)])?;
    let b = PackedSPD::from_vec(2, vec![c(2., 0.), c(0., 0.), c(3., 0.)])?;
    let e = a.generalized_selected_eigendecomposition(
        &b,
        GeneralizedEigenproblem::AxEqualsLambdaBx,
        EigenRange::Index { first: 1, last: 1 },
    )?;
    assert_eq!(e.count, 1);
    assert!((e.eigenvalues[0] - 3.).abs() < 1e-10);
    Ok(())
}
