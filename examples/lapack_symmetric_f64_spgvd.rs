use matrixpacked::{GeneralizedEigenproblem, PackedSPD, PackedSymmetric};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::from_vec(2, vec![4_f64, 0., 9.])?;
    let b = PackedSPD::from_vec(2, vec![2_f64, 0., 3.])?;
    let e = a.generalized_eigendecomposition_divide_conquer(
        &b,
        GeneralizedEigenproblem::AxEqualsLambdaBx,
    )?;
    assert!((e.eigenvalues[0] - 2.).abs() < 1e-10 && (e.eigenvalues[1] - 3.).abs() < 1e-10);
    Ok(())
}
