mod common;
use common::assert_slice_close;
use matrixpacked::{PackedUpper, PackedUpperView};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [2f32, 3f32, 4f32];
    let a = PackedUpperView::<f32>::from_slice(2, &storage)?;
    let x = [1f32, 2f32];
    let y = a.mul_vector(&x)?;
    assert_slice_close(&y, &[8f32, 8f32], 1e-4);
    Ok(())
}
