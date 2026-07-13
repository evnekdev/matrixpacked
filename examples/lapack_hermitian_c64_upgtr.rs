//! Demonstrates ZUPGTR generation of the unitary reduction matrix.
use matrixpacked::PackedHermitian;
use num_complex::Complex64;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex64::new;
    let r = PackedHermitian::from_vec(2, vec![c(2., 0.), c(1., -1.), c(3., 0.)])?
        .tridiagonal_reduction()?;
    let q = r.generate_q()?;
    assert_eq!(q.len(), 4);
    for col in 0..2 {
        let norm: f64 = (0..2).map(|row| q[row + col * 2].norm_sqr()).sum();
        assert!((norm - 1.).abs() < 1e-12);
    }
    Ok(())
}
