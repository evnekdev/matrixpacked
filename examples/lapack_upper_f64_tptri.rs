mod common;
use common::assert_slice_close;
use matrixpacked::PackedUpperViewMut;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = [2f64, 3f64, 4f64];
    let mut a = PackedUpperViewMut::<f64>::from_slice_mut(2, &mut storage)?;
    a.inverse_in_place()?;
    assert_slice_close(a.as_slice(), &[0.5f64, -0.375f64, 0.25f64], 1e-10);
    Ok(())
}
