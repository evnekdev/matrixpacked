mod common;
use common::assert_slice_close;
use num_complex::Complex32;
use matrixpacked::PackedUpperViewMut;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = [Complex32::new(2, 0), Complex32::new(3, 0), Complex32::new(4, 0)];
    let mut a = PackedUpperViewMut::<Complex32>::from_slice_mut(2, &mut storage)?;
    a.inverse_in_place()?;
    assert_slice_close(a.as_slice(), &[Complex32::new(0.5, 0), Complex32::new(-0.375, 0), Complex32::new(0.25, 0)], 1e-4);
    Ok(())
}
