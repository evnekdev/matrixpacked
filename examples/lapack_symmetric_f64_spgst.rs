use matrixpacked::{GeneralizedEigenproblem, PackedSPD, PackedSymmetric};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::from_vec(2, vec![4.0_f64, 0.0, 9.0])?;
    let b = PackedSPD::from_vec(2, vec![2.0_f64, 0.0, 3.0])?;
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
            assert!((lambda - expected[j]).abs() < 1e-12);
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
                GeneralizedEigenproblem::AxEqualsLambdaBx => a
                    .mul_vector(&x)?
                    .iter()
                    .zip(b.mul_vector(&x)?)
                    .map(|(&u, v)| u - lambda * v)
                    .collect(),
                GeneralizedEigenproblem::ABxEqualsLambdaX => a
                    .mul_vector(&b.mul_vector(&x)?)?
                    .iter()
                    .zip(&x)
                    .map(|(&u, &v)| u - lambda * v)
                    .collect(),
                GeneralizedEigenproblem::BAxEqualsLambdaX => b
                    .mul_vector(&ax)?
                    .iter()
                    .zip(&x)
                    .map(|(&u, &v)| u - lambda * v)
                    .collect(),
            };
            assert!(residual.into_iter().all(|r| r.abs() < 1e-11));
        }
    }
    Ok(())
}
