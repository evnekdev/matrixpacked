//! Demonstrates CPPSV, the one-shot HPD packed solve driver.
mod common;
use common::assert_slice_close;
use matrixpacked::PackedSPD;
use num_complex::Complex32;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex32::new;
    let a = PackedSPD::from_vec(2, vec![c(4.0, 0.0), c(1.0, -1.0), c(3.0, 0.0)])?;
    let b = [c(2.0, 1.0), c(1.0, -2.0)];
    let expected = a.cholesky()?.solve_vector(&b)?;
    assert_slice_close(&a.solve_once(&b, 1)?, &expected, 1e-5);
    Ok(())
}
