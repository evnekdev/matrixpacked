use matrixpacked::PackedSymmetric;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::from_vec(2, vec![2.0_f64, 1.0, 2.0])?;
    let result = a.eigendecomposition()?;
    assert!(result.eigenvalues.windows(2).all(|w| w[0] <= w[1]));
    let vectors = result.eigenvectors.unwrap();
    for (j, &lambda) in result.eigenvalues.iter().enumerate() {
        let v = &vectors[j * 2..(j + 1) * 2];
        let av = a.mul_vector(v)?;
        assert!((av[0] - lambda * v[0]).abs() < 1e-12);
        assert!((av[1] - lambda * v[1]).abs() < 1e-12);
        assert!((v.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() < 1e-12);
    }
    assert!((vectors[0] * vectors[2] + vectors[1] * vectors[3]).abs() < 1e-12);
    Ok(())
}
