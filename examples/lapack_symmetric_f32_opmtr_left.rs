//! Demonstrates left application with SOPMTR.
use matrixpacked::{ApplySide, OrthogonalOperation, PackedSymmetric};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let r = PackedSymmetric::from_vec(2, vec![2f32, 1., 3.])?.tridiagonal_reduction()?;
    let mut c = vec![1., 0., 0., 1.];
    r.apply_q_in_place(ApplySide::Left, OrthogonalOperation::None, 2, 2, 2, &mut c)?;
    assert_eq!(c.len(), 4);
    Ok(())
}
