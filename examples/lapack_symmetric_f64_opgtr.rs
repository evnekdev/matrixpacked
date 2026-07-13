//! Demonstrates DOPGTR generation of the orthogonal reduction matrix.
use matrixpacked::PackedSymmetric;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let r = PackedSymmetric::from_vec(3, vec![4f64,1.,2.,3.,0.5,2.])?.tridiagonal_reduction()?;
    let q = r.generate_q()?;
    assert_eq!(q.len(), 9);
    for col in 0..3 {
        let norm: f64 = (0..3).map(|row| q[row + col * 3] * q[row + col * 3]).sum();
        assert!((norm - 1.).abs() < 1e-12);
    }
    Ok(())
}
