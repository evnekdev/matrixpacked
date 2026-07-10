mod common;
use common::assert_slice_close;
use num_complex::Complex32;
use matrixpacked::PackedHermitian;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedHermitian::<Complex32>::from_vec(2, vec![Complex32::new(2, 0), Complex32::new(1, 1), Complex32::new(-1, 0)])?;
    let factor = a.factorize_in_place()?;
    assert_eq!(factor.dimension(), 2);
    Ok(())
}
