mod common;
use common::assert_slice_close;
use num_complex::Complex64;
use matrixpacked::PackedSPDView;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [Complex64::new(4, 0), Complex64::new(1, 1), Complex64::new(3, 0)];
    let a = PackedSPDView::<Complex64>::from_slice(2, &storage)?;
    let x = [Complex64::new(1, 0), Complex64::new(2, 0)];
    let y = a.mul_vector(&x)?;
    assert_slice_close(&y, &[Complex64::new(6, -2), Complex64::new(7, 1)], 1e-10);
    Ok(())
}
