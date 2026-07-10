mod common;
use common::assert_slice_close;
use num_complex::Complex64;
use matrixpacked::PackedLowerViewMut;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = [Complex64::new(2, 0), Complex64::new(3, 0), Complex64::new(4, 0)];
    let mut a = PackedLowerViewMut::<Complex64>::from_slice_mut(2, &mut storage)?;
    a.inverse_in_place()?;
    assert_slice_close(a.as_slice(), &[Complex64::new(0.5, 0), Complex64::new(-0.375, 0), Complex64::new(0.25, 0)], 1e-10);
    Ok(())
}
