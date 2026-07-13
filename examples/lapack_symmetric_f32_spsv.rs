//! Demonstrates SSPSV, the one-shot symmetric-indefinite packed solve driver.
mod common;
use common::assert_slice_close;
use matrixpacked::PackedSymmetric;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::from_vec(2, vec![0.0f32, 1.0, 0.0])?;
    let b = [2.0, 3.0];
    let expected = a.factorize()?.solve_vector(&b)?;
    assert_slice_close(&a.solve_once(&b, 1)?, &expected, 1e-5);
    Ok(())
}
