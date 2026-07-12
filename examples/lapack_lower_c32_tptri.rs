//! Demonstrates CTPTRI, the LAPACK packed-storage routine.
//! Computes the inverse of a nonsingular triangular matrix in packed storage, overwriting the stored matrix.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use num_complex::Complex32;
use matrixpacked::PackedLowerViewMut;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = [Complex32::new(2.0, 0.0), Complex32::new(3.0, 0.0), Complex32::new(4.0, 0.0)];
    let mut a = PackedLowerViewMut::<Complex32>::from_slice_mut(2, &mut storage)?;
    a.inverse_in_place()?;
    assert_slice_close(a.as_slice(), &[Complex32::new(0.5, 0.0), Complex32::new(-0.375, 0.0), Complex32::new(0.25, 0.0)], 1e-4);
    Ok(())
}
