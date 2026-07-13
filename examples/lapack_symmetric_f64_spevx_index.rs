use matrixpacked::{EigenRange, PackedSymmetric};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::from_vec(3, vec![1_f64, 0., 0., 2., 0., 3.])?;
    let e = a.selected_eigendecomposition(EigenRange::Index { first: 1, last: 2 })?;
    assert_eq!(e.count, 2);
    assert!(e.eigenvalues.windows(2).all(|w| w[0] <= w[1]));
    let z = e.eigenvectors.unwrap();
    for (j, &l) in e.eigenvalues.iter().enumerate() {
        let v = &z[j * 3..j * 3 + 3];
        let av = a.mul_vector(v)?;
        for i in 0..3 {
            assert!((av[i] - l * v[i]).abs() < 1e-12)
        }
        assert!((v.iter().map(|x| x * x).sum::<f64>() - 1.).abs() < 1e-12)
    }
    Ok(())
}
