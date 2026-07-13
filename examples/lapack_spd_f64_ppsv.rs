//! Demonstrates DPPSV, the one-shot SPD packed solve driver.
mod common;
use common::assert_slice_close;
use matrixpacked::PackedSPD;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSPD::from_vec(2, vec![4.0f64, 1.0, 3.0])?;
    let b = [6.0, 7.0];
    let expected = a.cholesky()?.solve_vector(&b)?;
    assert_slice_close(&a.solve_once(&b, 1)?, &expected, 1e-12);
    Ok(())
}
