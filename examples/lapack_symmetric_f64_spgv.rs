use matrixpacked::{GeneralizedEigenproblem, PackedSPD, PackedSymmetric};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::from_vec(2, vec![4_f64, 0., 9.])?;
    let b = PackedSPD::from_vec(2, vec![2_f64, 0., 3.])?;
    let e = a.generalized_eigendecomposition(&b, GeneralizedEigenproblem::AxEqualsLambdaBx)?;
    let z = e.eigenvectors.unwrap();
    for (j, &l) in e.eigenvalues.iter().enumerate() {
        let v = &z[j * 2..j * 2 + 2];
        let av = a.mul_vector(v)?;
        let bv = b.mul_vector(v)?;
        for i in 0..2 {
            assert!((av[i] - l * bv[i]).abs() < 1e-10)
        }
    }
    Ok(())
}
