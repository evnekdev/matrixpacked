mod common;
use common::assert_slice_close;
use num_complex::Complex32;
use matrixpacked::PackedHermitianViewMut;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = [Complex32::new(2, 0), Complex32::new(1, 1), Complex32::new(-1, 0)];
    let a = PackedHermitianViewMut::<Complex32>::from_slice_mut(2, &mut storage)?;
    let mut factor = a.factorize_in_place()?;
    factor.inverse_in_place()?;
    assert_slice_close(factor.as_slice(), &[Complex32::new(0.25, 0), Complex32::new(0.25, 0.25), Complex32::new(-0.5, 0)], 1e-4);
    Ok(())
}
