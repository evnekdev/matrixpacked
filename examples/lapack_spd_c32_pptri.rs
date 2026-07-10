mod common;
use common::assert_slice_close;
use num_complex::Complex32;
use matrixpacked::PackedSPDViewMut;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = [Complex32::new(4, 0), Complex32::new(1, 1), Complex32::new(3, 0)];
    let a = PackedSPDViewMut::<Complex32>::from_slice_mut(2, &mut storage)?;
    let mut factor = a.cholesky_in_place()?;
    factor.inverse_in_place()?;
    assert_slice_close(factor.as_slice(), &[Complex32::new(0.3, 0), Complex32::new(-0.1, -0.1), Complex32::new(0.4, 0)], 1e-4);
    Ok(())
}
