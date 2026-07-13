use matrixpacked::{EigenRange, PackedHermitian};
use num_complex::Complex64;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex64::new;
    let a = PackedHermitian::from_vec(2, vec![c(2., 0.), c(0., 1.), c(2., 0.)])?;
    let e = a.selected_eigendecomposition(EigenRange::Index { first: 0, last: 0 })?;
    assert_eq!(e.count, 1);
    let v = e.eigenvectors.unwrap();
    let av = a.mul_vector(&v)?;
    for i in 0..2 {
        assert!((av[i] - v[i] * e.eigenvalues[0]).norm() < 1e-12)
    }
    assert!((v.iter().map(|x| x.norm_sqr()).sum::<f64>() - 1.).abs() < 1e-12);
    Ok(())
}
