use matrixpacked::{GeneralizedEigenproblem, PackedSPD, PackedSymmetric};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::from_vec(2, vec![4.0_f32, 0.0, 9.0])?;
    let b = PackedSPD::from_vec(2, vec![2.0_f32, 0.0, 3.0])?;
    let factor = b.cholesky()?;
    let l = [factor.as_slice()[0], factor.as_slice()[2]];

    for problem in [
        GeneralizedEigenproblem::AxEqualsLambdaBx,
        GeneralizedEigenproblem::ABxEqualsLambdaX,
        GeneralizedEigenproblem::BAxEqualsLambdaX,
    ] {
        let expected = a.generalized_eigenvalues(&b, problem)?;
        let standard = a.generalized_reduction(&factor, problem)?;
        let eig = standard.eigendecomposition()?;
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
            let residual: Vec<f32> = match problem {
                GeneralizedEigenproblem::AxEqualsLambdaBx => {
                    let bx = b.mul_vector(&x)?;
                    ax.iter().zip(bx).map(|(&u, v)| u - lambda * v).collect()
                }
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
            assert!(residual.into_iter().all(|r: f32| r.abs() < 2e-4));
        }
    }
    Ok(())
}
