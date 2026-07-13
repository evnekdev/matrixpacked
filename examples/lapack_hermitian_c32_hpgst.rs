use matrixpacked::{GeneralizedEigenproblem, PackedHermitian, PackedSPD};
use num_complex::Complex32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex32::new;
    let a = PackedHermitian::from_vec(2, vec![c(4., 0.), c(0., 0.), c(9., 0.)])?;
    let b = PackedSPD::from_vec(2, vec![c(2., 0.), c(0., 0.), c(3., 0.)])?;
    let factor = b.cholesky()?;
    let l = [factor.as_slice()[0], factor.as_slice()[2]];

    for problem in [
        GeneralizedEigenproblem::AxEqualsLambdaBx,
        GeneralizedEigenproblem::ABxEqualsLambdaX,
        GeneralizedEigenproblem::BAxEqualsLambdaX,
    ] {
        let expected = a.generalized_eigenvalues(&b, problem)?;
        let eig = a
            .generalized_reduction(&factor, problem)?
            .eigendecomposition()?;
        for (j, &lambda) in eig.eigenvalues.iter().enumerate() {
            assert!((lambda - expected[j]).abs() < 2e-5);
            let y = &eig.eigenvectors.as_ref().unwrap()[j * 2..j * 2 + 2];
            let x: Vec<_> = y
                .iter()
                .zip(l)
                .map(|(&yi, li)| match problem {
                    GeneralizedEigenproblem::BAxEqualsLambdaX => li * yi,
                    _ => yi / li,
                })
                .collect();
            let ax = a.mul_vector(&x)?;
            let residual: Vec<_> = match problem {
                GeneralizedEigenproblem::AxEqualsLambdaBx => ax
                    .iter()
                    .zip(b.mul_vector(&x)?)
                    .map(|(&u, v)| u - v * lambda)
                    .collect(),
                GeneralizedEigenproblem::ABxEqualsLambdaX => a
                    .mul_vector(&b.mul_vector(&x)?)?
                    .iter()
                    .zip(&x)
                    .map(|(&u, &v)| u - v * lambda)
                    .collect(),
                GeneralizedEigenproblem::BAxEqualsLambdaX => b
                    .mul_vector(&ax)?
                    .iter()
                    .zip(&x)
                    .map(|(&u, &v)| u - v * lambda)
                    .collect(),
            };
            assert!(residual.into_iter().all(|r| r.norm() < 2e-4));
        }
    }
    Ok(())
}
