//! Demonstrates right conjugate-transpose application with ZUPMTR.
use matrixpacked::{ApplySide, PackedHermitian, UnitaryOperation};
use num_complex::Complex64;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex64::new;
    let r = PackedHermitian::from_vec(2, vec![c(2., 0.), c(1., -1.), c(3., 0.)])?
        .tridiagonal_reduction()?;
    let mut x = vec![c(1., 0.), c(0., 0.)];
    r.apply_q_in_place(
        ApplySide::Right,
        UnitaryOperation::ConjugateTranspose,
        1,
        2,
        1,
        &mut x,
    )?;
    Ok(())
}
