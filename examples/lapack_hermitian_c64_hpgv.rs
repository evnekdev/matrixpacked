use matrixpacked::{GeneralizedEigenproblem, PackedHermitian, PackedSPD};
use num_complex::Complex64;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex64::new;
    let a = PackedHermitian::from_vec(2, vec![c(4., 0.), c(0., 0.), c(9., 0.)])?;
    let b = PackedSPD::from_vec(2, vec![c(2., 0.), c(0., 0.), c(3., 0.)])?;
    let e = a.generalized_eigendecomposition(&b, GeneralizedEigenproblem::AxEqualsLambdaBx)?;
    let z = e.eigenvectors.unwrap();
    for (j, &l) in e.eigenvalues.iter().enumerate() {
        let v = &z[j * 2..j * 2 + 2];
        let av = a.mul_vector(v)?;
        let bv = b.mul_vector(v)?;
        for i in 0..2 {
            assert!((av[i] - bv[i] * l).norm() < 1e-10)
        }
    }
    Ok(())
}
