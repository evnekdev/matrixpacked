use matrixpacked::PackedSymmetric;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::from_vec(2, vec![2_f64, 1., 2.])?;
    let e = a.eigendecomposition_divide_conquer()?;
    assert!(e.eigenvalues.windows(2).all(|w| w[0] <= w[1]));
    let z = e.eigenvectors.unwrap();
    for (j, &l) in e.eigenvalues.iter().enumerate() {
        let v = &z[j * 2..j * 2 + 2];
        let av = a.mul_vector(v)?;
        assert!((av[0] - l * v[0]).abs() < 1e-12 && (av[1] - l * v[1]).abs() < 1e-12);
        assert!((v.iter().map(|x| x * x).sum::<f64>() - 1.).abs() < 1e-12)
    }
    assert!((z[0] * z[2] + z[1] * z[3]).abs() < 1e-12);
    Ok(())
}
