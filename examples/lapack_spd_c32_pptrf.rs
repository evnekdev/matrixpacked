mod common;
use common::assert_slice_close;
use num_complex::Complex32;
use matrixpacked::PackedSPD;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSPD::<Complex32>::from_vec(2, vec![Complex32::new(4, 0), Complex32::new(1, 1), Complex32::new(3, 0)])?;
    let factor = a.cholesky_in_place()?;
    assert_eq!(factor.dimension(), 2);
    Ok(())
}
