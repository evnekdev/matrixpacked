//! Demonstrates SSPTRD packed tridiagonal reduction.
use matrixpacked::PackedSymmetric;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let r =
        PackedSymmetric::from_vec(3, vec![4f32, 1., 2., 3., 0.5, 2.])?.tridiagonal_reduction()?;
    assert_eq!(r.diagonal().len(), 3);
    assert_eq!(r.off_diagonal().len(), 2);
    assert_eq!(r.tau().len(), 2);
    Ok(())
}
