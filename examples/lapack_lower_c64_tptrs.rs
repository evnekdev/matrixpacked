mod common;
use common::assert_slice_close;
use num_complex::Complex64;
use matrixpacked::PackedLower;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedLower::<Complex64>::from_vec(2, vec![Complex64::new(2, 0), Complex64::new(3, 0), Complex64::new(4, 0)])?;
    let mut b = [Complex64::new(2, 0), Complex64::new(11, 0)];
    a.solve_vector_in_place(&mut b)?;
    assert_slice_close(&b, &[Complex64::new(1, 0), Complex64::new(2, 0)], 1e-10);
    Ok(())
}
