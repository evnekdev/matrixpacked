use matrixpacked::{EigenRange, PackedHermitian};
use num_complex::Complex64;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex64::new;
    let a = PackedHermitian::from_vec(2, vec![c(2., 0.), c(0., 1.), c(2., 0.)])?;
    let e = a.selected_eigendecomposition(EigenRange::Value {
        lower: 1.,
        upper: 3.,
    })?;
    assert_eq!(e.count, 1);
    assert!((e.eigenvalues[0] - 3.).abs() < 1e-12);
    let v = e.eigenvectors.unwrap();
    let av = a.mul_vector(&v)?;
    for i in 0..2 {
        assert!((av[i] - v[i] * e.eigenvalues[0]).norm() < 1e-12)
    }
    Ok(())
}
