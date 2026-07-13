use matrixpacked::{GeneralizedEigenproblem, PackedSPD, PackedSymmetric};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::from_vec(2, vec![4_f32, 0., 9.])?;
    let b = PackedSPD::from_vec(2, vec![2_f32, 0., 3.])?;
    let v = a.generalized_eigenvalues(&b, GeneralizedEigenproblem::AxEqualsLambdaBx)?;
    assert!((v[0] - 2.).abs() < 1e-4 && (v[1] - 3.).abs() < 1e-4);
    Ok(())
}
