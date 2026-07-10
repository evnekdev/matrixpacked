mod common;
use common::assert_slice_close;
use num_complex::Complex64;
use matrixpacked::PackedHermitian;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedHermitian::<Complex64>::from_vec(2, vec![Complex64::new(2, 0), Complex64::new(1, 1), Complex64::new(-1, 0)])?;
    let factor = a.factorize_in_place()?;
    assert_eq!(factor.dimension(), 2);
    Ok(())
}
