mod common;
use common::assert_slice_close;
use matrixpacked::PackedSPDViewMut;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = [4f64, 1f64, 3f64];
    let a = PackedSPDViewMut::<f64>::from_slice_mut(2, &mut storage)?;
    let mut factor = a.cholesky_in_place()?;
    factor.inverse_in_place()?;
    assert_slice_close(factor.as_slice(), &[0.2727272727272727f64, -0.09090909090909091f64, 0.36363636363636365f64], 1e-10);
    Ok(())
}
