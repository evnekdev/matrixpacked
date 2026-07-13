use matrixpacked::{EigenRange, GeneralizedEigenproblem, PackedSPD, PackedSymmetric};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::from_vec(2, vec![4_f64, 0., 9.])?;
    let b = PackedSPD::from_vec(2, vec![2_f64, 0., 3.])?;
    let e = a.generalized_selected_eigendecomposition(
        &b,
        GeneralizedEigenproblem::AxEqualsLambdaBx,
        EigenRange::Index { first: 1, last: 1 },
    )?;
    assert_eq!(e.count, 1);
    assert!((e.eigenvalues[0] - 3.).abs() < 1e-10);
    Ok(())
}
