mod common;
use common::assert_slice_close;
use num_complex::Complex32;
use matrixpacked::PackedSymmetricViewMut;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = [Complex32::new(4, 0), Complex32::new(1, 0), Complex32::new(3, 0)];
    let a = PackedSymmetricViewMut::<Complex32>::from_slice_mut(2, &mut storage)?;
    let mut factor = a.factorize_in_place()?;
    factor.inverse_in_place()?;
    assert_slice_close(factor.as_slice(), &[Complex32::new(0.2727272727272727, 0), Complex32::new(-0.09090909090909091, 0), Complex32::new(0.36363636363636365, 0)], 1e-4);
    Ok(())
}
