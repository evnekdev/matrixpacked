//! Demonstrates ZTPTRI, the LAPACK packed-storage routine.
//! Computes the inverse of a nonsingular triangular matrix in packed storage, overwriting the stored matrix.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use num_complex::Complex64;
use matrixpacked::PackedUpperViewMut;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = [Complex64::new(2, 0), Complex64::new(3, 0), Complex64::new(4, 0)];
    let mut a = PackedUpperViewMut::<Complex64>::from_slice_mut(2, &mut storage)?;
    a.inverse_in_place()?;
    assert_slice_close(a.as_slice(), &[Complex64::new(0.5, 0), Complex64::new(-0.375, 0), Complex64::new(0.25, 0)], 1e-10);
    Ok(())
}
