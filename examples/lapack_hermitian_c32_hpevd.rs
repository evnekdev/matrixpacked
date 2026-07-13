use matrixpacked::PackedHermitian;
use num_complex::Complex32;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex32::new;
    let a = PackedHermitian::from_vec(2, vec![c(2., 0.), c(0., 1.), c(2., 0.)])?;
    let e = a.eigendecomposition_divide_conquer()?;
    assert!(e.eigenvalues.windows(2).all(|w| w[0] <= w[1]));
    let z = e.eigenvectors.unwrap();
    for (j, &l) in e.eigenvalues.iter().enumerate() {
        let v = &z[j * 2..j * 2 + 2];
        let av = a.mul_vector(v)?;
        assert!((av[0] - v[0] * l).norm() < 1e-5 && (av[1] - v[1] * l).norm() < 1e-5);
        assert!((v.iter().map(|x| x.norm_sqr()).sum::<f32>() - 1.).abs() < 1e-5)
    }
    assert!((z[0].conj() * z[2] + z[1].conj() * z[3]).norm() < 1e-5);
    Ok(())
}
