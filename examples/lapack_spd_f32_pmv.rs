mod common;
use common::assert_slice_close;
use matrixpacked::PackedSPDView;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [4f32, 1f32, 3f32];
    let a = PackedSPDView::<f32>::from_slice(2, &storage)?;
    let x = [1f32, 2f32];
    let y = a.mul_vector(&x)?;
    assert_slice_close(&y, &[6f32, 7f32], 1e-4);
    Ok(())
}
