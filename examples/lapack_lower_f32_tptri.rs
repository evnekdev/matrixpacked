//! Demonstrates STPTRI, the LAPACK packed-storage routine.
//! Computes the inverse of a nonsingular triangular matrix in packed storage, overwriting the stored matrix.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use matrixpacked::PackedLowerViewMut;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = [2f32, 3f32, 4f32];
    let mut a = PackedLowerViewMut::<f32>::from_slice_mut(2, &mut storage)?;
    a.inverse_in_place()?;
    assert_slice_close(a.as_slice(), &[0.5f32, -0.375f32, 0.25f32], 1e-4);
    Ok(())
}
