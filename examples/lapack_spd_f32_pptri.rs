//! Demonstrates SPPTRI, the LAPACK packed-storage routine.
//! Computes the inverse of a positive-definite matrix from its packed Cholesky factorization produced by `xPPTRF`.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use matrixpacked::PackedSPDViewMut;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = [4f32, 1f32, 3f32];
    let a = PackedSPDViewMut::<f32>::from_slice_mut(2, &mut storage)?;
    let mut factor = a.cholesky_in_place()?;
    factor.inverse_in_place()?;
    assert_slice_close(factor.as_slice(), &[0.2727272727272727f32, -0.09090909090909091f32, 0.36363636363636365f32], 1e-4);
    Ok(())
}
