mod common;
use common::assert_slice_close;
use num_complex::Complex32;
use matrixpacked::PackedLower;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedLower::<Complex32>::from_vec(2, vec![Complex32::new(2, 0), Complex32::new(3, 0), Complex32::new(4, 0)])?;
    let mut b = [Complex32::new(2, 0), Complex32::new(11, 0)];
    a.solve_vector_in_place(&mut b)?;
    assert_slice_close(&b, &[Complex32::new(1, 0), Complex32::new(2, 0)], 1e-4);
    Ok(())
}
