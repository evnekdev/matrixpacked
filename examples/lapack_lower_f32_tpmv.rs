//! Demonstrates STPMV, the BLAS packed-storage routine.
//! Computes `x := A*x` (or a transpose/conjugate-transpose variant) for a triangular matrix `A` stored in packed form.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use matrixpacked::PackedLowerView;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [2f32, 3f32, 4f32];
    let a = PackedLowerView::<f32>::from_slice(2, &storage)?;
    let x = [1f32, 2f32];
    let y = a.mul_vector(&x)?;
    assert_slice_close(&y, &[2f32, 11f32], 1e-4);
    Ok(())
}
