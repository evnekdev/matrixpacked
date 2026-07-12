//! Demonstrates SSPMV, the BLAS packed-storage routine.
//! Computes `y := alpha*A*x + beta*y` for a real symmetric matrix `A` stored in packed form.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use matrixpacked::PackedSymmetricView;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [4f32, 1f32, 3f32];
    let a = PackedSymmetricView::<f32>::from_slice(2, &storage)?;
    let x = [1f32, 2f32];
    let mut y = [1f32, 1f32];
    a.mul_vector_into(&x, &mut y, 1f32, 0f32)?;
    assert_slice_close(&y, &[6f32, 7f32], 1e-4);
    Ok(())
}
