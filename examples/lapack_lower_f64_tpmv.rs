mod common;
use common::assert_slice_close;
use matrixpacked::{PackedLower, PackedLowerView};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [2f64, 3f64, 4f64];
    let a = PackedLowerView::<f64>::from_slice(2, &storage)?;
    let x = [1f64, 2f64];
    let y = a.mul_vector(&x)?;
    assert_slice_close(&y, &[2f64, 11f64], 1e-10);
    Ok(())
}
