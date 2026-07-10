mod common;
use common::assert_slice_close;
use matrixpacked::PackedSymmetricView;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [4f64, 1f64, 3f64];
    let a = PackedSymmetricView::<f64>::from_slice(2, &storage)?;
    let x = [1f64, 2f64];
    let mut y = [1f64, 1f64];
    a.mul_vector_into(&x, &mut y, 1f64, 0f64)?;
    assert_slice_close(&y, &[6f64, 7f64], 1e-10);
    Ok(())
}
