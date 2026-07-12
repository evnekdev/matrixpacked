//! Demonstrates DSPMV, the BLAS packed-storage routine.
//! Computes `y := alpha*A*x + beta*y` for a real symmetric matrix `A` stored in packed form.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use matrixpacked::PackedSPDView;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [4f64, 1f64, 3f64];
    let a = PackedSPDView::<f64>::from_slice(2, &storage)?;
    let x = [1f64, 2f64];
    let y = a.mul_vector(&x)?;
    assert_slice_close(&y, &[6f64, 7f64], 1e-10);
    Ok(())
}
