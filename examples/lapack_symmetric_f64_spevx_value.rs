use matrixpacked::{EigenRange, PackedSymmetric};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::from_vec(3, vec![1_f64, 0., 0., 2., 0., 3.])?;
    let e = a.selected_eigendecomposition(EigenRange::Value {
        lower: 1.,
        upper: 2.,
    })?;
    assert_eq!(e.count, 1);
    assert!((e.eigenvalues[0] - 2.).abs() < 1e-12);
    let v = e.eigenvectors.unwrap();
    let av = a.mul_vector(&v)?;
    for i in 0..3 {
        assert!((av[i] - e.eigenvalues[0] * v[i]).abs() < 1e-12)
    }
    Ok(())
}
