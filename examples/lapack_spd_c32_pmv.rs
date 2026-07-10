mod common;
use common::assert_slice_close;
use num_complex::Complex32;
use matrixpacked::PackedSPDView;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [Complex32::new(4, 0), Complex32::new(1, 1), Complex32::new(3, 0)];
    let a = PackedSPDView::<Complex32>::from_slice(2, &storage)?;
    let x = [Complex32::new(1, 0), Complex32::new(2, 0)];
    let y = a.mul_vector(&x)?;
    assert_slice_close(&y, &[Complex32::new(6, -2), Complex32::new(7, 1)], 1e-4);
    Ok(())
}
